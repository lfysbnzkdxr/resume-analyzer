"""Base agent class with shared DeepSeek LLM setup."""

import os
from typing import Optional

from langchain_openai import ChatOpenAI
from src.core.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def get_llm(temperature: float = 0.0, api_key: Optional[str] = None):
    """Get a configured DeepSeek LLM instance (OpenAI-compatible API).

    Args:
        temperature: LLM temperature (0.0 = deterministic).
        api_key: DeepSeek API key. If None, reads from st.session_state
                 (Streamlit), then os.environ (eval/testing).

    Priority: explicit api_key > st.session_state > os.environ
    """
    if not api_key:
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
