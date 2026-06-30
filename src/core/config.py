"""Configuration management — API keys and app settings.

All settings can be overridden via RA_* environment variables.
Load .env once on first access (lazy), not at module import time.
"""

import os
from pathlib import Path


def _get_env(key: str, default: str) -> str:
    """Read .env on first call, then delegate to os.getenv."""
    if not _get_env._loaded:
        from dotenv import load_dotenv

        load_dotenv()
        _get_env._loaded = True
    return os.getenv(key, default)


_get_env._loaded = False

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOAD_DIR = DATA_DIR / "uploads"

# DeepSeek API
DEEPSEEK_API_KEY = _get_env("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _get_env("RA_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _get_env("RA_MODEL", "deepseek-chat")

# Embedding
EMBEDDING_MODEL = _get_env("RA_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DIM = 512

# RAG
CHUNK_SIZE = int(_get_env("RA_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(_get_env("RA_CHUNK_OVERLAP", "50"))
TOP_K_RETRIEVAL = int(_get_env("RA_TOP_K", "5"))

# API
API_TIMEOUT = int(_get_env("RA_TIMEOUT", "60"))

# HuggingFace mirror (China mainland)
HF_ENDPOINT = _get_env("RA_HF_ENDPOINT", "https://hf-mirror.com")

# App
APP_TITLE = "AI 简历智能分析"
