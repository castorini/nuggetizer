import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

# Attempt to import handlers, adjust path if necessary based on execution context
try:
    from src.nuggetizer.core.llm import LLMHandler
    from src.nuggetizer.core.async_llm import AsyncLLMHandler
    # Patching where 'OpenAI' and 'AsyncOpenAI' are LOOKED UP by the handlers
    OPENAI_CLIENT_PATCH_PATH = "src.nuggetizer.core.llm.OpenAI"
    ASYNC_OPENAI_CLIENT_PATCH_PATH = "src.nuggetizer.core.async_llm.AsyncOpenAI"
    GET_OPENROUTER_API_KEY_PATH = "src.nuggetizer.utils.api.get_openrouter_api_key"
    # Tiktoken is imported directly in llm.py and async_llm.py
    TIKTOKEN_PATCH_PATH = "tiktoken.get_encoding"
except ImportError:
    # Fallback for different execution context (e.g. running tests directly)
    from nuggetizer.core.llm import LLMHandler
    from nuggetizer.core.async_llm import AsyncLLMHandler
    OPENAI_CLIENT_PATCH_PATH = "nuggetizer.core.llm.OpenAI"
    ASYNC_OPENAI_CLIENT_PATCH_PATH = "nuggetizer.core.async_llm.AsyncOpenAI"
    GET_OPENROUTER_API_KEY_PATH = "nuggetizer.utils.api.get_openrouter_api_key"
    TIKTOKEN_PATCH_PATH = "tiktoken.get_encoding"


# Global mock for tiktoken.get_encoding
# We want to mock it once for all tests in this module as it's called in run() methods
mock_global_tiktoken_encoding = MagicMock()
mock_global_tiktoken_encoding.encode.return_value = [] # Simulate encoding returning a list of tokens

@patch(TIKTOKEN_PATCH_PATH, new=mock_global_tiktoken_encoding)
class TestLLMHandlerOpenRouter(unittest.TestCase):

    def setUp(self):
        # Patch get_openrouter_api_key specifically for this test class
        self.get_key_patcher = patch(GET_OPENROUTER_API_KEY_PATH)
        self.mock_get_openrouter_api_key = self.get_key_patcher.start()
        self.mock_get_openrouter_api_key.return_value = "test_openrouter_key_from_env"

        # Patch OpenAI client constructor
        self.openai_constructor_patcher = patch(OPENAI_CLIENT_PATCH_PATH)
        self.MockOpenAIConstructor = self.openai_constructor_patcher.start()

        # This is the mock for the *instance* of the OpenAI client
        self.mock_openai_client_instance = MagicMock()
        self.mock_openai_client_instance.chat.completions.create = MagicMock()
        self.MockOpenAIConstructor.return_value = self.mock_openai_client_instance

    def tearDown(self):
        self.get_key_patcher.stop()
        self.openai_constructor_patcher.stop()

    def test_openrouter_initialization_with_model_prefix(self):
        handler = LLMHandler(model="openrouter/some-model")
        self.mock_get_openrouter_api_key.assert_called_once()
        self.MockOpenAIConstructor.assert_called_once_with(
            api_key="test_openrouter_key_from_env",
            base_url="https://openrouter.ai/api/v1"
        )
        self.assertEqual(handler.api_type, "openrouter")
        self.assertEqual(handler.api_base, "https://openrouter.ai/api/v1")

    def test_openrouter_initialization_with_api_type(self):
        handler = LLMHandler(model="some-model", api_type="openrouter")
        self.mock_get_openrouter_api_key.assert_called_once()
        self.MockOpenAIConstructor.assert_called_once_with(
            api_key="test_openrouter_key_from_env",
            base_url="https://openrouter.ai/api/v1"
        )
        self.assertEqual(handler.api_type, "openrouter")

    def test_openrouter_initialization_with_custom_base_url(self):
        custom_url = "https://custom.openrouter.ai/api"
        handler = LLMHandler(model="openrouter/some-model", api_base=custom_url)
        self.mock_get_openrouter_api_key.assert_called_once() # Still called to get key if not provided
        self.MockOpenAIConstructor.assert_called_once_with(
            api_key="test_openrouter_key_from_env",
            base_url=custom_url
        )
        self.assertEqual(handler.api_base, custom_url)

    def test_openrouter_initialization_with_direct_api_key(self):
        direct_key = "direct_or_key_123"
        handler = LLMHandler(model="openrouter/some-model", api_keys=direct_key)
        self.mock_get_openrouter_api_key.assert_not_called()
        self.MockOpenAIConstructor.assert_called_once_with(
            api_key=direct_key,
            base_url="https://openrouter.ai/api/v1"
        )

    def test_openrouter_missing_api_key(self):
        self.mock_get_openrouter_api_key.return_value = None
        with self.assertRaisesRegex(ValueError, "OpenRouter API key not found"):
            LLMHandler(model="openrouter/some-model")

    def test_openrouter_run_method(self):
        handler = LLMHandler(model="openrouter/my-model") # Initializes client

        # Configure the mock response from client.chat.completions.create
        mock_completion_response = MagicMock()
        mock_completion_response.choices[0].message.content = "Test response"
        self.mock_openai_client_instance.chat.completions.create.return_value = mock_completion_response

        messages = [{"role": "user", "content": "Hello"}]
        response, token_count = handler.run(messages=messages)

        self.mock_openai_client_instance.chat.completions.create.assert_called_once_with(
            model="openrouter/my-model",
            messages=messages,
            temperature=0, # Default temperature
            max_completion_tokens=2048, # Default max_tokens from run method
            timeout=30 # Default timeout
        )
        self.assertEqual(response, "Test response")
        # token_count depends on the global tiktoken mock, which returns encode=[] -> len 0
        self.assertEqual(token_count, 0)

    def test_openrouter_api_key_rotation(self):
        api_keys = ["key1", "key2"]
        handler = LLMHandler(model="openrouter/my-model", api_type="openrouter", api_keys=api_keys)

        self.assertEqual(handler.api_keys, api_keys)
        self.assertEqual(handler.client.api_key, "key1")

        # Mock create to fail first time, succeed second time
        self.mock_openai_client_instance.chat.completions.create.side_effect = [
            Exception("Simulated API error on key1"), # First call fails
            MagicMock(choices=[MagicMock(message=MagicMock(content="Success on key2"))]) # Second call succeeds
        ]

        # First call to run - should fail with key1, then rotate
        messages = [{"role": "user", "content": "Test"}]
        response, _ = handler.run(messages=messages)

        self.assertEqual(self.mock_openai_client_instance.chat.completions.create.call_count, 2)
        self.assertEqual(handler.current_key_idx, 1) # Key index should have advanced
        self.assertEqual(handler.client.api_key, "key2") # Client api_key property updated
        self.assertEqual(response, "Success on key2")


@patch(TIKTOKEN_PATCH_PATH, new=mock_global_tiktoken_encoding)
class TestAsyncLLMHandlerOpenRouter(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self): # Renamed for clarity, unittest calls this if defined
        self.get_key_patcher = patch(GET_OPENROUTER_API_KEY_PATH)
        self.mock_get_openrouter_api_key = self.get_key_patcher.start()
        self.mock_get_openrouter_api_key.return_value = "test_async_openrouter_key_from_env"

        self.async_openai_constructor_patcher = patch(ASYNC_OPENAI_CLIENT_PATCH_PATH)
        self.MockAsyncOpenAIConstructor = self.async_openai_constructor_patcher.start()

        self.mock_async_openai_client_instance = AsyncMock() # Use AsyncMock for async methods
        self.mock_async_openai_client_instance.chat.completions.create = AsyncMock() # create is async
        self.MockAsyncOpenAIConstructor.return_value = self.mock_async_openai_client_instance

    async def asyncTearDown(self):
        self.get_key_patcher.stop()
        self.async_openai_constructor_patcher.stop()

    async def test_openrouter_initialization_with_model_prefix_async(self):
        handler = AsyncLLMHandler(model="openrouter/some-async-model")
        self.mock_get_openrouter_api_key.assert_called_once()
        self.MockAsyncOpenAIConstructor.assert_called_once_with(
            api_key="test_async_openrouter_key_from_env",
            base_url="https://openrouter.ai/api/v1"
        )
        self.assertEqual(handler.api_type, "openrouter")
        self.assertEqual(handler.api_base, "https://openrouter.ai/api/v1")

    async def test_openrouter_initialization_with_api_type_async(self):
        handler = AsyncLLMHandler(model="some-async-model", api_type="openrouter")
        self.mock_get_openrouter_api_key.assert_called_once()
        self.MockAsyncOpenAIConstructor.assert_called_once_with(
            api_key="test_async_openrouter_key_from_env",
            base_url="https://openrouter.ai/api/v1"
        )
        self.assertEqual(handler.api_type, "openrouter")

    async def test_openrouter_initialization_with_custom_base_url_async(self):
        custom_url = "https://custom.async.openrouter.ai/api"
        handler = AsyncLLMHandler(model="openrouter/some-async-model", api_base=custom_url)
        self.mock_get_openrouter_api_key.assert_called_once()
        self.MockAsyncOpenAIConstructor.assert_called_once_with(
            api_key="test_async_openrouter_key_from_env",
            base_url=custom_url
        )
        self.assertEqual(handler.api_base, custom_url)

    async def test_openrouter_initialization_with_direct_api_key_async(self):
        direct_key = "direct_async_or_key_456"
        handler = AsyncLLMHandler(model="openrouter/some-async-model", api_keys=direct_key)
        self.mock_get_openrouter_api_key.assert_not_called()
        self.MockAsyncOpenAIConstructor.assert_called_once_with(
            api_key=direct_key,
            base_url="https://openrouter.ai/api/v1"
        )

    async def test_openrouter_missing_api_key_async(self):
        self.mock_get_openrouter_api_key.return_value = None
        with self.assertRaisesRegex(ValueError, "OpenRouter API key not found"):
            AsyncLLMHandler(model="openrouter/some-async-model")

    async def test_openrouter_run_method_async(self):
        handler = AsyncLLMHandler(model="openrouter/my-async-model")

        mock_completion_response = MagicMock() # AsyncMock isn't strictly needed for the response object itself
        mock_completion_response.choices[0].message.content = "Async test response"
        # The method on the client that produces the response needs to be an AsyncMock
        self.mock_async_openai_client_instance.chat.completions.create.return_value = mock_completion_response

        messages = [{"role": "user", "content": "Hello Async"}]
        response, token_count = await handler.run(messages=messages)

        self.mock_async_openai_client_instance.chat.completions.create.assert_awaited_once_with(
            model="openrouter/my-async-model",
            messages=messages,
            temperature=0,
            max_completion_tokens=2048,
            timeout=30
        )
        self.assertEqual(response, "Async test response")
        self.assertEqual(token_count, 0) # Due to global tiktoken mock

    async def test_openrouter_api_key_rotation_async(self):
        api_keys = ["async_key1", "async_key2"]
        handler = AsyncLLMHandler(model="openrouter/my-async-model", api_type="openrouter", api_keys=api_keys)

        self.assertEqual(handler.api_keys, api_keys)
        # Initial client should be set up with the first key
        self.MockAsyncOpenAIConstructor.assert_called_with(api_key="async_key1", base_url="https://openrouter.ai/api/v1")
        # Direct check of client.api_key if possible, or rely on re-initialization with new key.
        # The current implementation re-initializes the client on key rotation. Let's test that.

        # Mock create to fail first time, succeed second time
        # First, the client instance that was created in __init__
        current_client_instance = self.MockAsyncOpenAIConstructor.return_value

        # Store the new AsyncMock instance that will be created after key rotation
        new_mock_async_client_instance = AsyncMock()
        new_mock_async_client_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Success on async_key2"))])
        )

        # Side effect for the constructor: first return the initial client, then the new one
        self.MockAsyncOpenAIConstructor.side_effect = [
            current_client_instance, # Original instance
            new_mock_async_client_instance # New instance after rotation
        ]

        # Side effect for the create method of the *first* client instance
        current_client_instance.chat.completions.create.side_effect = Exception("Simulated API error on async_key1")

        messages = [{"role": "user", "content": "Test Async"}]
        response, _ = await handler.run(messages=messages)

        # Check that the first client's create was called
        current_client_instance.chat.completions.create.assert_awaited_once()

        # Constructor called twice: once in __init__, once after key rotation
        self.assertEqual(self.MockAsyncOpenAIConstructor.call_count, 2)
        # Check that the second constructor call used the rotated key
        self.MockAsyncOpenAIConstructor.assert_any_call(api_key="async_key2", base_url="https://openrouter.ai/api/v1")

        self.assertEqual(handler.current_key_idx, 1)
        self.assertEqual(handler.client, new_mock_async_client_instance) # Client instance was replaced
        self.assertEqual(response, "Success on async_key2")
        # Verify the new client's create method was called
        new_mock_async_client_instance.chat.completions.create.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
