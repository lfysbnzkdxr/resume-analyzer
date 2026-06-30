"""Base agent class with shared DeepSeek LLM setup."""

from langchain_openai import ChatOpenAI
from src.core.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, API_TIMEOUT


def get_llm(api_key: str, temperature: float = 0.0):
    """Get a configured DeepSeek LLM instance (OpenAI-compatible API).

    Args:
        api_key: DeepSeek API key (required).
        temperature: LLM temperature (0.0 = deterministic).
    """
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=temperature,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=API_TIMEOUT,
        max_retries=2,
    )
