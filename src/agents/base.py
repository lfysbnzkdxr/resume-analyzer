"""Base agent class with shared DeepSeek LLM setup."""

import os

from langchain_openai import ChatOpenAI
from src.core.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def get_llm(temperature: float = 0.0):
    """Get a configured DeepSeek LLM instance (OpenAI-compatible API).

    Reads the API key from st.session_state first (Streamlit per-user isolation),
    falling back to os.environ for non-Streamlit contexts (eval, testing).
    """
    try:
        import streamlit as st
        api_key = st.session_state.get("DEEPSEEK_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=temperature,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=60,          # 单次请求超时（秒），避免无限挂起
        max_retries=2,       # HTTP 层面重试（LangChain 内置 tenacity）
    )
