"""Single analysis page — upload a resume PDF + paste JD text, run analysis."""

import streamlit as st
import os
from pathlib import Path

from src.core.orchestrator import run_single_analysis
from src.ui.components.score_chart import display_radar_chart, display_suggestions


def render():
    st.header("📄 单份简历分析")
    st.markdown("上传一份简历 PDF，粘贴目标职位描述，获取 AI 匹配评估报告。")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("上传简历 (PDF)", type=["pdf"], key="single_pdf")
        pdf_path = None
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")
            tmp_dir = Path("data/uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = str(tmp_dir / uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    with col2:
        jd_text = st.text_area(
            "粘贴职位描述 (JD)",
            height=250,
            placeholder="请粘贴职位描述文本...",
        )

    if st.button("▶ 开始分析", type="primary", use_container_width=True):
        if not uploaded_file:
            st.error("请上传简历 PDF")
            return
        if not jd_text.strip():
            st.error("请输入职位描述")
            return

        with st.spinner("AI 分析中...（简历提取 → JD 分析 → 匹配评估）"):
            try:
                result = run_single_analysis(pdf_path, jd_text)
                _display_result(result)
            except Exception as e:
                st.error(f"分析过程出错: {str(e)}")
            finally:
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)


def _display_result(result):
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
