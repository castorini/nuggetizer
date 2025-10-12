import time
from typing import Any, Dict, List, Optional, Tuple, Union

import tiktoken
from openai import AzureOpenAI, OpenAI

from ..utils.api import (
    get_azure_openai_args,
    get_openai_api_key,
    get_openrouter_api_key,
    get_vllm_api_key,
)


class LLMHandler:
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
    ):
        self.model = model
        self.context_size = context_size

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
                "1. OpenAI API key (OPEN_AI_API_KEY environment variable)\n"
                "2. OpenRouter API key (OPENROUTER_API_KEY environment variable)\n"
                "3. Azure OpenAI credentials (AZURE_OPENAI_API_KEY, etc.)\n"
                "4. Use vLLM local server (use_vllm=True)\n"
                "5. Pass api_keys parameter directly to Nuggetizer constructor"
            )

        self.api_keys = [api_keys] if isinstance(api_keys, str) else api_keys
        self.current_key_idx = 0
        self.vllm_port = vllm_port
        assert api_type is not None
        self.client = self._initialize_client(api_type, api_base, api_version)

    def _initialize_client(
        self, api_type: str, api_base: str | None, api_version: str | None
    ) -> Union[AzureOpenAI, OpenAI]:
        if api_type == "azure" and all([api_base, api_version]):
            assert api_base is not None
            return AzureOpenAI(
                api_key=self.api_keys[0],
                api_version=api_version,
                azure_endpoint=api_base,
            )
        elif api_type == "openai":
            return OpenAI(api_key=self.api_keys[0])
        elif api_type == "openrouter":
            return OpenAI(api_key=self.api_keys[0], base_url=api_base)
        elif api_type == "vllm":
            full_url = api_base
            return OpenAI(api_key=self.api_keys[0], base_url=full_url)
        else:
            raise ValueError(f"Invalid API type: {api_type}")

    def run(
        self, messages: List[Dict[str, str]], temperature: float = 0
    ) -> Tuple[str, int, Optional[Dict[str, Any]], Optional[str]]:
        """
        Run LLM inference and return content, token count, usage metadata, and reasoning.

        Returns:
            Tuple of (content, token_count, usage_metadata, reasoning_content)
        """

        remaining_retry = 5
        while remaining_retry > 0:
            if "o1" in self.model or "o3" in self.model or "o4" in self.model:
                # System message is not supported for o1 models
                new_messages = messages[1:]
                new_messages[0]["content"] = (
                    messages[0]["content"] + "\n" + messages[1]["content"]
                )
                messages = new_messages[:]
            if (
                "o1" in self.model
                or "o3" in self.model
                or "o4" in self.model
                or "gpt-5" in self.model
            ):
                temperature = 1.0
            try:
                completion = None
                # Use different parameters for vLLM vs other APIs
                completion_params: Dict[str, Any]
                if hasattr(self.client, "base_url") and "localhost" in str(
                    self.client.base_url
                ):
                    # vLLM specific parameters
                    completion_params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": 4096,
                        "timeout": 60,
                    }
                else:
                    # Standard OpenAI/other APIs
                    completion_params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_completion_tokens": 4096,
                        "timeout": 60,
                    }
                completion = self.client.chat.completions.create(**completion_params)
                # print(f"🔍 DEBUG LLM: API call completed successfully") # not
                # removed because it's very helpful for debugging

                response = completion.choices[0].message.content
                # print(f"🔍 DEBUG LLM: Full response: {completion}") # not
                # removed because it's very helpful for debugging

                # Extract reasoning content if available
                reasoning_content = None
                message = completion.choices[0].message
                # Check for reasoning field in the message
                if hasattr(message, "reasoning") and message.reasoning:
                    reasoning_content = message.reasoning
                elif (
                    hasattr(message, "reasoning_content") and message.reasoning_content
                ):
                    reasoning_content = message.reasoning_content
                # Also check if it's a dict with reasoning field
                elif (
                    isinstance(message, dict)
                    and "reasoning" in message
                    and message["reasoning"]
                ):
                    reasoning_content = message["reasoning"]
                elif (
                    isinstance(message, dict)
                    and "reasoning_content" in message
                    and message["reasoning_content"]
                ):
                    reasoning_content = message["reasoning_content"]
                else:
                    print(f"No reasoning found in response from {self.model}")

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
                        "total_tokens": getattr(completion.usage, "total_tokens", None),
                    }

                try:
                    # For newer models like gpt-4o that may not have specific
                    # encodings yet
                    if "gpt-4o" in self.model or "gpt-4.1" in self.model:
                        encoding = tiktoken.get_encoding("o200k_base")
                    elif (
                        "qwen" in self.model.lower()
                        or "qwen2" in self.model.lower()
                        or "qwen3" in self.model.lower()
                    ):
                        # Use cl100k_base for Qwen models as they typically use
                        # similar tokenization
                        encoding = tiktoken.get_encoding("cl100k_base")
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
                print(f"LLM Inference Error: {str(e)}")
                remaining_retry -= 1
                if remaining_retry <= 0:
                    raise RuntimeError("Reached max of 5 retries.")
                # Don't retry in case of safety trigger.
                if (
                    completion
                    and completion.choices
                    and completion.choices[0].finish_reason == "content_filter"
                ):
                    raise ValueError("Request blocked by content filter.")
                if self.api_keys is not None:
                    self.current_key_idx = (self.current_key_idx + 1) % len(
                        self.api_keys
                    )
                    self.client.api_key = self.api_keys[self.current_key_idx]
                time.sleep(0.1)

        # This should never be reached, but mypy requires it
        raise RuntimeError("Unexpected end of method")
