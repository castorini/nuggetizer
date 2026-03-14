import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI

from ..utils.api import (
    get_azure_openai_args,
    get_openai_api_key,
    get_openrouter_api_key,
    get_vllm_api_key,
)


class AsyncLLMHandler:
    SUPPORTED_REASONING_EFFORTS = (
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    )

    def __init__(
        self,
        model: str,
        api_keys: Optional[Union[str, List[str]]] = None,
        context_size: int = 8192,
        api_type: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        use_azure_openai: bool = False,
        use_openrouter: bool = False,
        use_vllm: bool = False,
        openrouter_api_key: Optional[str] = None,
        vllm_port: int = 8000,
        reasoning_effort: Optional[str] = None,
    ):
        self.model = model
        self.context_size = context_size
        if (
            reasoning_effort is not None
            and reasoning_effort not in self.SUPPORTED_REASONING_EFFORTS
        ):
            raise ValueError(
                "Unsupported reasoning_effort: "
                f"{reasoning_effort}. Expected one of "
                f"{', '.join(self.SUPPORTED_REASONING_EFFORTS)}."
            )
        self.reasoning_effort = reasoning_effort

        # Auto-configure API keys and Azure settings if not provided
        if use_azure_openai and (
            api_keys is None
            or (api_type == "azure" or (api_type is None and "gpt" in model.lower()))
        ):
            azure_args = get_azure_openai_args()
            api_type = "azure"
            api_base = azure_args.get("api_base")
            api_version = azure_args.get("api_version")
            api_keys = azure_args.get("api_key", api_keys) or get_openai_api_key()
        else:
            # Check for explicit API keys first, then environment variables
            if api_keys is None:
                if use_vllm:
                    # Use vLLM local server
                    api_keys = get_vllm_api_key()
                    api_base = f"http://localhost:{vllm_port}/v1"
                    api_type = "vllm"
                elif use_openrouter:
                    # Use OpenRouter API
                    openrouter_key = openrouter_api_key or get_openrouter_api_key()
                    if openrouter_key is not None:
                        api_keys = openrouter_key
                        api_base = "https://openrouter.ai/api/v1"
                        api_type = "openrouter"
                    else:
                        raise ValueError(
                            "use_openrouter=True but no OpenRouter API key found"
                        )
                else:
                    # Try OpenAI API key first, then OpenRouter as fallback
                    openai_key = get_openai_api_key()
                    if openai_key is not None:
                        api_keys = openai_key
                        api_type = "openai"
                    else:
                        # Try OpenRouter API key as fallback
                        openrouter_key = openrouter_api_key or get_openrouter_api_key()
                        if openrouter_key is not None:
                            api_keys = openrouter_key
                            api_base = "https://openrouter.ai/api/v1"
                            api_type = "openrouter"
            else:
                # Use provided API keys with OpenAI by default
                api_type = "openai"
        # Ensure we have a valid API type and keys
        if api_type is None or api_keys is None:
            raise ValueError(
                "No valid API key found. Please provide either:\n"
                "1. OpenAI API key (OPENAI_API_KEY environment variable)\n"
                "2. OpenRouter API key (OPENROUTER_API_KEY environment variable)\n"
                "3. Azure OpenAI credentials (AZURE_OPENAI_API_KEY, etc.)\n"
                "4. Use vLLM local server (use_vllm=True)\n"
                "5. Pass api_keys parameter directly to Nuggetizer constructor"
            )

        self.api_keys = [api_keys] if isinstance(api_keys, str) else api_keys
        self.current_key_idx = 0
        assert api_type is not None
        self.client = self._initialize_client(api_type, api_base, api_version)

    def _initialize_client(
        self, api_type: str, api_base: str | None, api_version: str | None
    ) -> Union[AsyncAzureOpenAI, AsyncOpenAI]:
        if api_type == "azure" and all([api_base, api_version]):
            assert api_base is not None
            return AsyncAzureOpenAI(
                api_key=self.api_keys[0],
                api_version=api_version,
                azure_endpoint=api_base,
            )
        elif api_type == "openai":
            return AsyncOpenAI(api_key=self.api_keys[0])
        elif api_type == "openrouter":
            return AsyncOpenAI(api_key=self.api_keys[0], base_url=api_base)
        elif api_type == "vllm":
            full_url = api_base
            return AsyncOpenAI(api_key=self.api_keys[0], base_url=full_url)
        else:
            raise ValueError(f"Invalid API type: {api_type}")

    def _build_reasoning_params(self) -> Dict[str, Any]:
        """Build provider-specific reasoning request parameters."""
        if self.reasoning_effort is None:
            return {}
        if hasattr(self.client, "base_url") and "openrouter.ai" in str(
            self.client.base_url
        ):
            return {
                "extra_body": {
                    "reasoning": {
                        "effort": self.reasoning_effort,
                        "exclude": False,
                    }
                }
            }
        return {"reasoning_effort": self.reasoning_effort}

    def _uses_reasoning_style_api(self) -> bool:
        return (
            "o1" in self.model
            or "o3" in self.model
            or "o4" in self.model
            or "gpt-5" in self.model
        )

    def _uses_responses_reasoning_api(self) -> bool:
        return self.reasoning_effort is not None and self._uses_reasoning_style_api()

    @staticmethod
    def _normalize_messages(
        messages: List[Dict[str, str]], model: str
    ) -> List[Dict[str, str]]:
        if ("o1" in model or "o3" in model or "o4" in model) and len(messages) >= 2:
            normalized_messages = [message.copy() for message in messages[1:]]
            normalized_messages[0]["content"] = (
                messages[0]["content"] + "\n" + messages[1]["content"]
            )
            return normalized_messages
        return messages

    @classmethod
    def _build_responses_input(
        cls, messages: List[Dict[str, str]], model: str
    ) -> List[Dict[str, Any]]:
        normalized_messages = cls._normalize_messages(messages, model)
        return [
            {
                "type": "message",
                "role": message["role"],
                "content": [
                    {
                        "type": "input_text",
                        "text": message["content"],
                    }
                ],
            }
            for message in normalized_messages
        ]

    def _extract_responses_text_and_reasoning(
        self, response: Any
    ) -> Tuple[str, Optional[str]]:
        text = ""
        if hasattr(response, "output_text") and response.output_text:
            text = str(response.output_text)
        else:
            for item in getattr(response, "output", []):
                item_type = getattr(item, "type", None)
                if item_type is None and isinstance(item, dict):
                    item_type = item.get("type")
                if item_type != "message":
                    continue
                content_items = getattr(item, "content", None)
                if content_items is None and isinstance(item, dict):
                    content_items = item.get("content", [])
                for content in content_items or []:
                    content_type = getattr(content, "type", None)
                    if content_type is None and isinstance(content, dict):
                        content_type = content.get("type")
                    if content_type == "output_text":
                        content_text = getattr(content, "text", None)
                        if content_text is None and isinstance(content, dict):
                            content_text = content.get("text")
                        if content_text:
                            text = str(content_text)

        reasoning_parts: List[str] = []
        for item in getattr(response, "output", []):
            item_type = getattr(item, "type", None)
            if item_type is None and isinstance(item, dict):
                item_type = item.get("type")
            if item_type != "reasoning":
                continue
            direct_parts: List[str] = []
            summaries = getattr(item, "summary", None)
            if summaries is None and isinstance(item, dict):
                summaries = item.get("summary", [])
            summary_parts: List[str] = []
            for summary in summaries or []:
                if isinstance(summary, str):
                    summary_parts.append(summary)
                    continue
                summary_type = getattr(summary, "type", None)
                if summary_type is None and isinstance(summary, dict):
                    summary_type = summary.get("type")
                if summary_type not in (None, "summary_text"):
                    continue
                summary_text = getattr(summary, "text", None)
                if summary_text is None and isinstance(summary, dict):
                    summary_text = summary.get("text")
                if summary_text:
                    summary_parts.append(str(summary_text))
            direct_reasoning = getattr(item, "reasoning", None)
            if direct_reasoning is None and isinstance(item, dict):
                direct_reasoning = item.get("reasoning")
            if direct_reasoning:
                direct_parts.append(str(direct_reasoning))
            direct_reasoning_content = getattr(item, "reasoning_content", None)
            if direct_reasoning_content is None and isinstance(item, dict):
                direct_reasoning_content = item.get("reasoning_content")
            if direct_reasoning_content:
                direct_parts.append(str(direct_reasoning_content))
            item_content = getattr(item, "content", None)
            if item_content is None and isinstance(item, dict):
                item_content = item.get("content", [])
            for content in item_content or []:
                if isinstance(content, str):
                    direct_parts.append(content)
                    continue
                content_text = getattr(content, "text", None)
                if content_text is None and isinstance(content, dict):
                    content_text = content.get("text")
                if content_text:
                    direct_parts.append(str(content_text))
            if hasattr(self.client, "base_url") and "openrouter.ai" in str(
                self.client.base_url
            ):
                reasoning_parts.extend(direct_parts or summary_parts)
            else:
                reasoning_parts.extend(summary_parts or direct_parts)

        reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
        return text, reasoning

    @staticmethod
    def _extract_reasoning_content(message: Any) -> Optional[str]:
        """Extract reasoning text from common OpenAI-compatible response shapes."""
        if hasattr(message, "reasoning") and message.reasoning:
            return str(message.reasoning)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            return str(message.reasoning_content)
        if isinstance(message, dict):
            if "reasoning" in message and message["reasoning"]:
                return str(message["reasoning"])
            if "reasoning_content" in message and message["reasoning_content"]:
                return str(message["reasoning_content"])
        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            if "reasoning" in model_extra and model_extra["reasoning"]:
                return str(model_extra["reasoning"])
            if "reasoning_content" in model_extra and model_extra["reasoning_content"]:
                return str(model_extra["reasoning_content"])
        return None

    async def run(
        self, messages: List[Dict[str, str]], temperature: float = 0
    ) -> Tuple[str, int, Optional[Dict[str, Any]], Optional[str]]:
        """
        Run async LLM inference and return content, token count, usage metadata, and reasoning.

        Returns:
            Tuple of (content, token_count, usage_metadata, reasoning_content)
        """
        while True:
            if (
                "o1" in self.model
                or "o3" in self.model
                or "o4" in self.model
                or "gpt-5" in self.model
            ):
                temperature = 1.0
            try:
                normalized_messages = self._normalize_messages(messages, self.model)
                if self._uses_responses_reasoning_api():
                    assert self.reasoning_effort is not None
                    response_obj = await cast(Any, self.client.responses).create(
                        model=self.model,
                        input=self._build_responses_input(messages, self.model),
                        max_output_tokens=2048,
                        timeout=30,
                        reasoning={
                            "effort": self.reasoning_effort,
                            "summary": "auto",
                        },
                    )
                    response, reasoning_content = (
                        self._extract_responses_text_and_reasoning(response_obj)
                    )
                    usage_metadata = None
                    usage = getattr(response_obj, "usage", None)
                    if usage:
                        usage_metadata = {
                            "prompt_tokens": getattr(
                                usage,
                                "input_tokens",
                                getattr(usage, "prompt_tokens", None),
                            ),
                            "completion_tokens": getattr(
                                usage,
                                "output_tokens",
                                getattr(usage, "completion_tokens", None),
                            ),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        }
                else:
                    completion_params: Dict[str, Any] = {
                        "model": self.model,
                        "messages": normalized_messages,
                        "temperature": temperature,
                        "max_completion_tokens": 2048,
                        "timeout": 30,
                    }
                    completion_params.update(self._build_reasoning_params())
                    completion = await self.client.chat.completions.create(
                        **completion_params
                    )
                    response = completion.choices[0].message.content

                    # Extract reasoning content if available
                    message = completion.choices[0].message
                    reasoning_content = self._extract_reasoning_content(message)

                    # Handle None response
                    if response is None:
                        response = ""

                    # Extract usage metadata
                    usage_metadata = None
                    if hasattr(completion, "usage") and completion.usage:
                        usage_metadata = {
                            "prompt_tokens": getattr(
                                completion.usage, "prompt_tokens", None
                            ),
                            "completion_tokens": getattr(
                                completion.usage, "completion_tokens", None
                            ),
                            "total_tokens": getattr(
                                completion.usage, "total_tokens", None
                            ),
                        }

                try:
                    # For newer models like gpt-4o that may not have specific
                    # encodings yet
                    if "gpt-4o" in self.model:
                        encoding = tiktoken.get_encoding("o200k_base")
                    else:
                        encoding = tiktoken.get_encoding(self.model)
                except Exception:
                    encoding = tiktoken.get_encoding("cl100k_base")

                # Ensure response is a string before encoding
                response_str = str(response) if response is not None else ""
                return (
                    response_str,
                    len(encoding.encode(response_str)),
                    usage_metadata,
                    reasoning_content,
                )
            except Exception as e:
                print(f"Error: {str(e)}")
                if self.api_keys is not None:
                    self.current_key_idx = (self.current_key_idx + 1) % len(
                        self.api_keys
                    )
                    self.client.api_key = self.api_keys[self.current_key_idx]
                time.sleep(0.1)
