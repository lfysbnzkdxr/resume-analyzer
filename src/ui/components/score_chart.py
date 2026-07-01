"""Score visualization components using Streamlit native charts."""

import streamlit as st

from src.ui.theme import score_color


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
            st.markdown(
                f"<p style='color:{color};font-size:24px;font-weight:bold;'>{score:.0f}</p>",
                unsafe_allow_html=True,
            )
            if dim.get("details"):
                with st.expander("详情"):
                    st.write(dim["details"])


def display_analysis_result(result):
    """Display a full analysis result: score chart, suggestions, summary."""
    st.divider()
    st.subheader("📊 分析结果")

    dims = [d.model_dump() for d in result.dimensions]
    suggs = [s.model_dump() for s in result.suggestions]

    display_radar_chart(dims, result.overall_score)

    if suggs:
        st.divider()
        st.subheader("💡 改进建议")
        display_suggestions(suggs)

    st.divider()
    st.subheader("📝 总结")
    st.write(result.summary or "（暂无总结）")


def _score_color(score: float) -> str:
    return score_color(score)


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
            f"<p style='color:{color};margin:4px 0;'>{icon} [{label}] {s.get('content', '')}</p>",
            unsafe_allow_html=True,
        )
