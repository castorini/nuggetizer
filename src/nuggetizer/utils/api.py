import os
from typing import Dict

from dotenv import load_dotenv


def get_openai_api_key() -> str:
    load_dotenv(dotenv_path=".env")
    openai_api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return openai_api_key


def get_azure_openai_args() -> Dict[str, str]:
    load_dotenv(dotenv_path=".env")
    azure_args = {
        "api_type": "azure",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "api_base": os.getenv("AZURE_OPENAI_API_BASE"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    }

    # Sanity check
    assert all(
        list(azure_args.values())
    ), "Ensure that `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION` are set"
    return azure_args


def get_cohere_api_key() -> str:
    load_dotenv(dotenv_path=".env.local")
    return os.getenv("CO_API_KEY")


def get_anyscale_api_key() -> str:
    load_dotenv(dotenv_path=".env.local")
    return os.getenv("ANYSCALE_API_KEY")
