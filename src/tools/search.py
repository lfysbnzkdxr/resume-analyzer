"""Resume library search tool for LangChain agents."""

from src.rag.retriever import retrieve_resumes_for_jd


def search_resumes_for_jd(jd_text: str, top_k: int = 5) -> list[dict]:
    """Search the library for resumes matching a job description."""
    return retrieve_resumes_for_jd(jd_text, top_k=top_k)
