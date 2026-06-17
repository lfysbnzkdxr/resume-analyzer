"""Library Agent — manages the resume library (add, list, delete, search)."""

from src.rag.loader import load_pdf
from src.rag.splitter import split_text
from src.rag.vector_store import add_resume, list_resumes, delete_resume, get_resume_count
from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP


def add_resume_to_library(file_path: str) -> dict:
    """Parse a PDF, chunk it, and add to the vector store."""
    text = load_pdf(file_path)
    if not text.strip():
        return {"success": False, "error": "PDF 内容为空或无法解析", "chunks": 0}

    chunks = split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    filename = file_path.replace("\\", "/").split("/")[-1]
    count = add_resume(filename, chunks)
    return {
        "success": True,
        "filename": filename,
        "chunks": count,
        "text_length": len(text),
    }


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
