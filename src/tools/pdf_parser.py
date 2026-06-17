"""PDF parsing tool for LangChain agents."""

from pathlib import Path
from src.rag.loader import load_pdf_with_metadata


def parse_resume_pdf(file_path: str) -> str:
    """Tool: parse a PDF resume and return its full text content."""
    path = Path(file_path)
    if not path.exists():
        return f"错误：文件不存在 {file_path}"
    if path.suffix.lower() != ".pdf":
        return "错误：仅支持 PDF 文件"

    try:
        result = load_pdf_with_metadata(str(path))
        text = result["text"]
        info = f"文件名: {result['filename']}\n页数: {result['page_count']}\n\n"
        return info + text
    except Exception as e:
        return f"PDF 解析失败: {str(e)}"
