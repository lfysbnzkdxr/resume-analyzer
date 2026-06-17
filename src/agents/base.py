"""Base agent class with shared DeepSeek LLM setup."""

from langchain_openai import ChatOpenAI
from src.core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def get_llm(temperature: float = 0.0):
    """Get a configured DeepSeek LLM instance (OpenAI-compatible API)."""
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=temperature,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
