"""High-level retriever combining search and formatting."""

from src.rag.vector_store import search_resumes, search_resumes_aggregated
from src.core.config import TOP_K_RETRIEVAL


def retrieve_relevant_context(query: str, top_k: int = TOP_K_RETRIEVAL) -> str:
    """Search resume library and format results as context string for the LLM."""
    results = search_resumes(query, top_k=top_k)

    if not results:
        return "（未在简历库中找到相关记录）"

    parts = []
    for r in results:
        filename = r["metadata"].get("filename", "unknown")
        parts.append(f"[来自: {filename} | 匹配度: {r['score']:.2f}]")
        parts.append(r["text"])
        parts.append("---")

    return "\n".join(parts)


def retrieve_resumes_for_jd(jd_text: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Search resumes matching a JD and return chunk-level ranked results."""
    return search_resumes(jd_text, top_k=top_k)


def retrieve_resumes_aggregated(jd_text: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Search resumes matching a JD, aggregated to unique resumes.

    Each result contains: filename, score, best_chunk, matching_chunks.
    """
    return search_resumes_aggregated(jd_text, top_k=top_k)
