"""High-level retriever combining search and formatting."""

from src.core.config import TOP_K_RETRIEVAL
from src.rag.vector_store import search_resumes, search_resumes_aggregated


def retrieve_resumes_for_jd(jd_text: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Search resumes matching a JD and return chunk-level ranked results."""
    return search_resumes(jd_text, top_k=top_k)


def retrieve_resumes_aggregated(jd_text: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Search resumes matching a JD, aggregated to unique resumes.

    Each result contains: filename, score, best_chunk, matching_chunks.
    """
    return search_resumes_aggregated(jd_text, top_k=top_k)
