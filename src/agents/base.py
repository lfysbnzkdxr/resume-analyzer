"""Base agent class with shared DeepSeek LLM setup."""

import os

from langchain_openai import ChatOpenAI
from src.core.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def get_llm(temperature: float = 0.0):
    """Get a configured DeepSeek LLM instance (OpenAI-compatible API).

    Reads the API key from os.environ at call time so Streamlit sidebar input
    (which sets os.environ["DEEPSEEK_API_KEY"]) takes effect without restart.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=temperature,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=60,          # 单次请求超时（秒），避免无限挂起
        max_retries=2,       # HTTP 层面重试（LangChain 内置 tenacity）
    )
