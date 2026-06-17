"""PDF document loader using PyMuPDF."""

from pathlib import Path
import fitz  # PyMuPDF


def load_pdf(file_path):
    """Extract all text from a PDF file."""
    doc = fitz.open(str(file_path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def load_pdf_with_metadata(file_path):
    """Extract text and basic metadata from a PDF."""
    doc = fitz.open(str(file_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    doc.close()
    full_text = "\n\n".join(p["text"] for p in pages)
    return {
        "filename": Path(file_path).name,
        "page_count": len(pages),
        "text": full_text,
        "pages": pages,
    }
