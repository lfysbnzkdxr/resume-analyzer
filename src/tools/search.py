"""Resume library search tool for LangChain agents."""

from src.rag.retriever import retrieve_relevant_context, retrieve_resumes_for_jd


def search_resume_library(query: str, top_k: int = 5) -> str:
    """Tool: search the resume library by semantic similarity and return formatted results."""
    return retrieve_relevant_context(query, top_k=top_k)


def search_resumes_for_jd(jd_text: str, top_k: int = 5) -> list[dict]:
    """Search the library for resumes matching a job description."""
    return retrieve_resumes_for_jd(jd_text, top_k=top_k)
