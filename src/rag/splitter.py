"""Text splitting strategies optimized for Chinese resumes.

Detects common resume section headers and uses them as natural chunk
boundaries, so each chunk contains semantically complete content
(e.g. a full "工作经验" section rather than an arbitrary 500-char slice).
"""

import re
from typing import List

# Common resume section header patterns (Chinese + English)
SECTION_HEADERS = re.compile(
    r"^\s*("
    r"(?:个人)?(?:信息|总结|简介|概况|概要|资料)"
    r"|基本资料"
    r"|自我评价"
    r"|求职意向"
    r"|技能(?:专长|清单|列表)?"
    r"|工作(?:经验|经历)"
    r"|项目(?:经验|经历)"
    r"|实习(?:经验|经历)"
    r"|教育(?:背景|经历)"
    r"|证书|荣誉|获奖"
    r"|语言(?:能力)?"
    r"|兴趣爱好"
    r"|Summary|Skills|Experience|Education"
    r"|Projects?|Certifications"
    r")[\s：:：]+",
    re.IGNORECASE | re.MULTILINE,
)


def _find_section_boundaries(text: str) -> List[int]:
    """Find all section start positions (character indices)."""
    boundaries = [0]  # first section starts at 0
    for m in SECTION_HEADERS.finditer(text):
        # Only treat as boundary if not at the very start (first section already covered)
        if m.start() > 0:
            boundaries.append(m.start())
    return sorted(set(boundaries))


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into chunks, using resume section headers as natural boundaries.

    Small sections (个人信息, 个人总结, 技能) are merged into a single chunk.
    Large sections (工作经验, 项目经验) exceeding *chunk_size* are sub-split
    using the original sentence-boundary strategy.
    """
    if not text.strip():
        return [text.strip()] if text.strip() else []

    boundaries = _find_section_boundaries(text)

    # Extract sections as (start, end) pairs
    sections = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        sec_text = text[start:end].strip()
        if sec_text:
            sections.append(sec_text)

    if not sections:
        return [text.strip()]

    # Merge small consecutive sections until the merged result
    # reaches at least chunk_size//2
    merged = []
    buffer = ""

    for sec in sections:
        # If this section alone is larger than chunk_size, flush buffer first
        if len(sec) > chunk_size:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.extend(_sub_split(sec, chunk_size, chunk_overlap))
            continue

        # If adding this section would exceed chunk_size, flush buffer then start new
        if buffer and len(buffer) + len(sec) + 1 > chunk_size:
            merged.append(buffer.strip())
            buffer = sec
        else:
            buffer = (buffer + "\n" + sec) if buffer else sec

    if buffer:
        merged.append(buffer.strip())

    return merged if merged else [text.strip()]


def _sub_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Fallback: split a single long text using sentence boundaries (original logic)."""
    separators = ["\n\n", "\n", "。", ".", " ", ""]
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            best_break = -1
            for sep in separators:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos > best_break:
                    best_break = pos
            if best_break > 0:
                sep_idx = separators.index(sep) if best_break > 0 else 0
                end = best_break + (len(sep) if sep_idx < len(separators) - 1 else 0)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap if end < text_len else end

    return chunks if chunks else [text.strip()]
