"""Text splitting strategies optimized for Chinese resumes."""

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split text into overlapping chunks, respecting Chinese sentence boundaries."""
    separators = ["\n\n", "\n", "。", ".", " ", ""]
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to find a good break point near the end boundary
        if end < text_len:
            best_break = -1
            for sep in separators:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos > best_break:
                    best_break = pos
            if best_break > 0:
                end = best_break + (len(sep) if separators.index(sep) < len(separators) - 1 else 0)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap if end < text_len else end

    return chunks if chunks else [text.strip()]
