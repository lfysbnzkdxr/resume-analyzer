"""Score visualization components using Streamlit native charts."""

import streamlit as st


def display_radar_chart(dimensions: list[dict], overall_score: float):
    """Display a radar-like chart using Streamlit's native progress bars."""
    st.subheader(f"综合匹配度: {overall_score:.0f}/100")

    cols = st.columns(2)
    for i, dim in enumerate(dimensions):
        col = cols[i % 2]
        with col:
            score = dim.get("score", 0)
            weight = dim.get("weight", 0)
            color = _score_color(score)
            st.markdown(f"**{dim.get('name', '')}** (权重 {weight:.0%})")
            st.progress(score / 100)
            st.markdown(f"<p style='color:{color};font-size:24px;font-weight:bold;'>{score:.0f}</p>",
                        unsafe_allow_html=True)
            if dim.get("details"):
                with st.expander("详情"):
                    st.write(dim["details"])


def _score_color(score: float) -> str:
    if score >= 85:
        return "#22c55e"
    elif score >= 65:
        return "#eab308"
    else:
        return "#ef4444"


def display_suggestions(suggestions: list[dict]):
    """Display improvement suggestions with priority colors."""
    priority_map = {
        "high": ("🔴", "#ef4444", "高优先级"),
        "medium": ("🟡", "#eab308", "中优先级"),
        "low": ("🟢", "#22c55e", "低优先级"),
    }
    for s in suggestions:
        icon, color, label = priority_map.get(s.get("priority", "low"), ("⚪", "#888", "未知"))
        st.markdown(
            f"<p style='color:{color};margin:4px 0;'>{icon} "
            f"[{label}] {s.get('content', '')}</p>",
            unsafe_allow_html=True,
        )
