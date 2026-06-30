"""PDF document loader using PyMuPDF."""

from pathlib import Path
from typing import Any, Union

import fitz  # PyMuPDF


def load_pdf(file_path: Union[str, Path]) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(str(file_path))
    pages: list[str] = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def load_pdf_with_metadata(file_path: Union[str, Path]) -> dict[str, Any]:
    """Extract text and basic metadata from a PDF."""
    doc = fitz.open(str(file_path))
    pages_list: list[dict[str, Any]] = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages_list.append({"page": i + 1, "text": text})
    doc.close()
    full_text = "\n\n".join(p["text"] for p in pages_list)
    return {
        "filename": Path(file_path).name,
        "page_count": len(pages_list),
        "text": full_text,
        "pages": pages_list,
    }
