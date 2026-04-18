import os

from dotenv import load_dotenv


def get_openai_api_key() -> str | None:
    load_dotenv(dotenv_path=".env")
    return os.getenv("OPENAI_API_KEY") or None


def get_azure_openai_args() -> dict[str, str | None]:
    load_dotenv(dotenv_path=".env")
    azure_args = {
        "api_type": "azure",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "api_base": os.getenv("AZURE_OPENAI_API_BASE"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    }

    # Sanity check
    assert all(list(azure_args.values())), (
        "Ensure that `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION` are set"
    )
    for key, value in azure_args.items():
        if value is None:
            raise ValueError(f"{key} not found in environment variables")
    else:
        return azure_args


def get_openrouter_api_key() -> str | None:
    load_dotenv(dotenv_path=".env")
    return os.getenv("OPENROUTER_API_KEY") or None


def get_vllm_api_key() -> str | None:
    """vLLM doesn't require authentication for local use, but we return a placeholder."""
    return "EMPTY"
