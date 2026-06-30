"""Library Manager — manages the resume library (add, list, delete, search).

This module is a thin application-layer wrapper around the RAG pipeline.
It does NOT involve any LLM/Agent calls — it only orchestrates PDF loading,
text splitting, and vector store operations.
"""

from pathlib import Path

from src.rag.loader import load_pdf
from src.rag.splitter import split_text
from src.rag.vector_store import add_resume, list_resumes, delete_resume, get_resume_count, rebuild_bm25_index
from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP


def add_resume_to_library(file_path: str, rebuild: bool = True) -> dict:
    """Parse a PDF, chunk it, and add to the vector store.

    Args:
        file_path: Path to the PDF file.
        rebuild: Rebuild BM25 index after adding.
                 Set to False during batch uploads, then call
                 rebuild_library_index() once at the end.
    """
    text = load_pdf(file_path)
    if not text.strip():
        return {"success": False, "error": "PDF 内容为空或无法解析", "chunks": 0}

    chunks = split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    filename = Path(file_path).name
    count = add_resume(filename, chunks, rebuild=rebuild)
    return {
        "success": True,
        "filename": filename,
        "chunks": count,
        "text_length": len(text),
    }


def rebuild_library_index():
    """Rebuild the BM25 index explicitly after batch uploads."""
    rebuild_bm25_index()


def list_library_resumes() -> list[str]:
    """List all resumes in the library."""
    return list_resumes()


def remove_resume_from_library(filename: str) -> int:
    """Remove a resume from the library by filename."""
    return delete_resume(filename)


def get_library_stats() -> dict:
    """Get library statistics."""
    return {
        "resume_count": get_resume_count(),
    }
