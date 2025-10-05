import time
from typing import Dict, List, Optional, Union, Tuple
import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI

from ..utils.api import (
    get_azure_openai_args,
    get_openai_api_key,
    get_openrouter_api_key,
    get_vllm_api_key,
)


class AsyncLLMHandler:
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
        assert api_type is not None
        assert api_base is not None
        assert api_version is not None
        self.client = self._initialize_client(api_type, api_base, api_version)

    def _initialize_client(
        self, api_type: str, api_base: str, api_version: str
    ) -> Union[AsyncAzureOpenAI, AsyncOpenAI]:
        if api_type == "azure" and all([api_base, api_version]):
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
            return AsyncOpenAI(api_key=self.api_keys[0], base_url=api_base)
        else:
            raise ValueError(f"Invalid API type: {api_type}")

    async def run(
        self, messages: List[Dict[str, str]], temperature: float = 0.0
    ) -> Tuple[str, int]:
        while True:
            if "o1" in self.model or "o3" in self.model:
                # System message is not supported for o1 models
                new_messages = messages[1:]
                new_messages[0]["content"] = (
                    messages[0]["content"] + "\n" + messages[1]["content"]
                )
                messages = new_messages[:]
                temperature = 1.0
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_completion_tokens=2048,
                    timeout=30,
                )
                response = completion.choices[0].message.content
                
                # Handle None response
                if response is None:
                    response = ""
                
                try:
                    # For newer models like gpt-4o that may not have specific encodings yet
                    if "gpt-4o" in self.model:
                        encoding = tiktoken.get_encoding("o200k_base")
                    else:
                        encoding = tiktoken.get_encoding(self.model)
                except Exception:
                    encoding = tiktoken.get_encoding("cl100k_base")
                return response, len(encoding.encode(response))
            except Exception as e:
                print(f"Error: {str(e)}")
                if self.api_keys is not None:
                    self.current_key_idx = (self.current_key_idx + 1) % len(
                        self.api_keys
                    )
                    self.client.api_key = self.api_keys[self.current_key_idx]
                time.sleep(0.1)
