"""Local embedding model wrapper using fastembed (pure ONNX, no torch conflicts)."""

import os
from fastembed import TextEmbedding

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_embedder = None


def get_embedding_model():
    """Lazy-load the embedding model (loaded once, cached)."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(
            model_name="BAAI/bge-small-zh-v1.5",
            max_length=512,
        )
    return _embedder


def embed_text(text: str) -> list[float]:
    """Embed a single text string into a vector."""
    model = get_embedding_model()
    vec = next(model.embed([text]))
    return vec.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts into vectors."""
    model = get_embedding_model()
    return [v.tolist() for v in model.embed(texts)]
