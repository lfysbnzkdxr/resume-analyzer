"""Suggestion card component."""

import streamlit as st


def display_suggestion_card(suggestion: dict):
    """Display a single suggestion in a card-like container."""
    priority = suggestion.get("priority", "low")
    category = suggestion.get("category", "")
    content = suggestion.get("content", "")

    labels = {
        "skill_gap": "技能缺口",
        "presentation": "表达优化",
        "format": "格式建议",
        "experience": "经验补充",
    }
    category_label = labels.get(category, category)

    border_colors = {"high": "#ef4444", "medium": "#eab308", "low": "#22c55e"}
    border = border_colors.get(priority, "#888")

    st.markdown(
        f"""<div style="border-left:4px solid {border};padding:8px 12px;margin:8px 0;
                    background:#f9fafb;border-radius:4px;">
            <strong>{category_label}</strong><br>
            {content}
        </div>""",
        unsafe_allow_html=True,
    )
