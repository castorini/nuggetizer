import time
from typing import Dict, List, Optional, Union, Tuple
import tiktoken
from openai import AzureOpenAI, OpenAI
from ..utils.api import get_azure_openai_args, get_openai_api_key, get_openrouter_api_key

class LLMHandler:
    def __init__(
        self,
        model: str,
        api_keys: Optional[Union[str, List[str]]] = None,
        context_size: int = 8192,
        api_type: Optional[str] = None, # "openai", "azure", "openrouter"
        api_base: Optional[str] = None, # Base URL for API
        api_version: Optional[str] = None, # API version, primarily for Azure
        use_azure_openai: bool = False, # Kept for backward compatibility, api_type="azure" is preferred
    ):
        self.model = model
        self.context_size = context_size
        self.api_version = api_version
        self.api_base = api_base
        
        resolved_api_type = api_type
        resolved_api_keys = api_keys

        # 1. Determine API type
        if resolved_api_type: # User explicitly provided api_type
            pass
        elif model.startswith("openrouter/"):
            resolved_api_type = "openrouter"
        elif use_azure_openai and ("gpt" in model.lower() or api_type == "azure"): # existing azure condition
            resolved_api_type = "azure"
        else:
            resolved_api_type = "openai" # Default

        self.api_type = resolved_api_type

        # 2. Configure based on API type
        if self.api_type == "openrouter":
            self.api_base = api_base or "https://openrouter.ai/api/v1"
            if resolved_api_keys:
                self.api_keys = [resolved_api_keys] if isinstance(resolved_api_keys, str) else resolved_api_keys
            else:
                openrouter_key = get_openrouter_api_key()
                if not openrouter_key:
                    raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable or pass api_keys.")
                self.api_keys = [openrouter_key]
            self.use_azure_openai = False # Ensure Azure is not accidentally used

        elif self.api_type == "azure":
            if not self.api_base or not self.api_version: # Check if essential Azure args are present
                azure_args = get_azure_openai_args() # This function raises error if not set
                self.api_base = self.api_base or azure_args.get("api_base")
                self.api_version = self.api_version or azure_args.get("api_version")
                # Key priority: user-provided -> azure_args -> get_openai_api_key (as fallback)
                resolved_api_keys = resolved_api_keys or azure_args.get("api_key") or get_openai_api_key()

            if not resolved_api_keys:
                 raise ValueError("Azure OpenAI API key not found.")
            self.api_keys = [resolved_api_keys] if isinstance(resolved_api_keys, str) else resolved_api_keys
            self.use_azure_openai = True

        elif self.api_type == "openai":
            if resolved_api_keys:
                self.api_keys = [resolved_api_keys] if isinstance(resolved_api_keys, str) else resolved_api_keys
            else:
                openai_key = get_openai_api_key()
                if not openai_key:
                    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass api_keys.")
                self.api_keys = [openai_key]
            self.use_azure_openai = False
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")

        if not self.api_keys or not self.api_keys[0]: # Ensure api_keys list is not empty and first key is valid
            raise ValueError(f"API key for {self.api_type} could not be resolved.")

        self.current_key_idx = 0
        self.client = self._initialize_client()
        
    def _initialize_client(self):
        if self.api_type == "azure":
            if not self.api_base or not self.api_version:
                raise ValueError("Azure API base or version not set for Azure client.")
            return AzureOpenAI(
                api_key=self.api_keys[self.current_key_idx],
                api_version=self.api_version,
                azure_endpoint=self.api_base
            )
        elif self.api_type == "openai":
            return OpenAI(api_key=self.api_keys[self.current_key_idx])
        elif self.api_type == "openrouter":
            return OpenAI(
                api_key=self.api_keys[self.current_key_idx],
                base_url=self.api_base
            )
        else:
            raise ValueError(f"Invalid API type for client initialization: {self.api_type}")

    def run(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0
    ) -> Tuple[str, int]:
        while True:
            if "o1" in self.model:
                # System message is not supported for o1 models
                new_messages = messages[1:]
                new_messages[0]["content"] = messages[0]["content"] + "\n" + messages[1]["content"]
                messages = new_messages[:]
                temperature = 1.0
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=2048,
                    timeout=30
                )
                response = completion.choices[0].message.content
                try:
                    # For newer models like gpt-4o that may not have specific encodings yet
                    if "gpt-4o" in self.model:
                        encoding = tiktoken.get_encoding("o200k_base")
                    else:
                        encoding = tiktoken.get_encoding(self.model)
                except Exception as e:
                    print(f"Error: {str(e)}")
                    encoding = tiktoken.get_encoding("cl100k_base")
                return response, len(encoding.encode(response))
            except Exception as e:
                print(f"Error: {str(e)}")
                if self.api_keys is not None:
                    self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                    self.client.api_key = self.api_keys[self.current_key_idx]
                time.sleep(0.1)
