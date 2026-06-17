"""PDF parsing tool for LangChain agents."""

from pathlib import Path
from langchain_core.tools import tool
from src.rag.loader import load_pdf_with_metadata


@tool
def parse_resume_pdf(file_path: str) -> str:
    """Tool: parse a PDF resume and return its full text content."""
    path = Path(file_path)
    if not path.exists():
        return f"错误：文件不存在 {file_path}"
    if path.suffix.lower() != ".pdf":
        return "错误：仅支持 PDF 文件"

    # Try PyMuPDF first
    try:
        result = load_pdf_with_metadata(str(path))
        text = result["text"]
        if text.strip():
            info = f"文件名: {result['filename']}\n页数: {result['page_count']}\n\n"
            return info + text
    except Exception:
        pass

    # Fallback: read as raw text (for test files, eval, etc.)
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if raw.strip():
            return f"文件名: {path.name}\n(文本回退模式)\n\n{raw}"
    except Exception:
        pass

    return f"错误：无法解析文件 {file_path}"
