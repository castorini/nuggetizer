import time
from typing import Any, cast

from openai import AsyncAzureOpenAI, AsyncOpenAI

from ..utils.api import (
    get_azure_openai_args,
    get_openai_api_key,
    get_openrouter_api_key,
    get_vllm_api_key,
)
from ._llm_shared import (
    SUPPORTED_REASONING_EFFORTS,
    build_responses_input,
    build_usage_metadata,
    extract_reasoning_content,
    extract_responses_text_and_reasoning,
    get_response_encoding,
    is_openrouter_client,
    normalize_messages,
    resolve_api_settings,
    uses_reasoning_style_api,
    uses_responses_reasoning_api,
    validate_reasoning_effort,
)


class AsyncLLMHandler:
    SUPPORTED_REASONING_EFFORTS = SUPPORTED_REASONING_EFFORTS

    def __init__(
        self,
        model: str,
        api_keys: str | list[str] | None = None,
        context_size: int = 8192,
        api_type: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        use_azure_openai: bool = False,
        use_openrouter: bool = False,
        use_vllm: bool = False,
        openrouter_api_key: str | None = None,
        vllm_port: int = 8000,
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.context_size = context_size
        validate_reasoning_effort(reasoning_effort)
        self.reasoning_effort = reasoning_effort
        api_type, api_base, api_version, self.api_keys = resolve_api_settings(
            model=model,
            api_keys=api_keys,
            api_type=api_type,
            api_base=api_base,
            api_version=api_version,
            use_azure_openai=use_azure_openai,
            use_openrouter=use_openrouter,
            use_vllm=use_vllm,
            openrouter_api_key=openrouter_api_key,
            vllm_port=vllm_port,
            get_azure_openai_args=get_azure_openai_args,
            get_openai_api_key=get_openai_api_key,
            get_openrouter_api_key=get_openrouter_api_key,
            get_vllm_api_key=get_vllm_api_key,
        )
        self.current_key_idx = 0
        self.client = self._initialize_client(api_type, api_base, api_version)

    def _initialize_client(
        self, api_type: str, api_base: str | None, api_version: str | None
    ) -> AsyncAzureOpenAI | AsyncOpenAI:
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

    def _build_reasoning_params(self) -> dict[str, Any]:
        """Build provider-specific reasoning request parameters."""
        if self.reasoning_effort is None:
            return {}
        if is_openrouter_client(self.client):
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
        return uses_reasoning_style_api(self.model)

    def _uses_responses_reasoning_api(self) -> bool:
        return uses_responses_reasoning_api(self.model, self.reasoning_effort)

    @staticmethod
    def _normalize_messages(
        messages: list[dict[str, str]], model: str
    ) -> list[dict[str, str]]:
        return normalize_messages(messages, model)

    @classmethod
    def _build_responses_input(
        cls, messages: list[dict[str, str]], model: str
    ) -> list[dict[str, Any]]:
        return build_responses_input(messages, model)

    def _extract_responses_text_and_reasoning(
        self, response: Any
    ) -> tuple[str, str | None]:
        return extract_responses_text_and_reasoning(
            response,
            prefer_direct_reasoning=is_openrouter_client(self.client),
        )

    @staticmethod
    def _extract_reasoning_content(message: Any) -> str | None:
        return extract_reasoning_content(message)

    async def run(
        self, messages: list[dict[str, str]], temperature: float = 0
    ) -> tuple[str, int, dict[str, Any] | None, str | None]:
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
                    usage_metadata = build_usage_metadata(
                        getattr(response_obj, "usage", None)
                    )
                else:
                    completion_params: dict[str, Any] = {
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
                    usage_metadata = build_usage_metadata(
                        getattr(completion, "usage", None)
                    )

                encoding = get_response_encoding(self.model)

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
