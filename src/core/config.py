"""Configuration management — API keys and app settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOAD_DIR = DATA_DIR / "uploads"

# DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Embedding
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512

# RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# App
APP_TITLE = "AI 简历智能分析"
