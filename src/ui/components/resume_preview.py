"""Resume text preview component."""

import streamlit as st


def display_resume_preview(text: str, max_lines: int = 50):
    """Show a scrollable preview of resume text."""
    lines = text.split("\n")
    if len(lines) > max_lines:
        preview = "\n".join(lines[:max_lines])
        preview += f"\n\n... (共 {len(lines)} 行，省略 {len(lines) - max_lines} 行)"
    else:
        preview = text

    with st.expander("📄 简历原文预览", expanded=False):
        st.text(preview)
