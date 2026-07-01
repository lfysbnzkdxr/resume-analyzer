"""Single analysis page — upload a resume PDF + paste JD text, run analysis."""

import logging
import os

import streamlit as st

from src.core.config import MAX_UPLOAD_SIZE_MB, UPLOAD_DIR
from src.core.orchestrator import run_single_analysis
from src.ui.components.score_chart import display_analysis_result

logger = logging.getLogger(__name__)


def render():
    st.header("📄 单份简历分析")
    st.markdown("上传一份简历 PDF，粘贴目标职位描述，获取 AI 匹配评估报告。")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("上传简历 (PDF)", type=["pdf"], key="single_pdf")
        pdf_path = None
        if uploaded_file:
            if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                st.error(f"文件过大（{uploaded_file.size / 1024 / 1024:.1f}MB），限制 {MAX_UPLOAD_SIZE_MB}MB")
                uploaded_file = None
                st.stop()
            st.success(f"✅ {uploaded_file.name}")
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            pdf_path = str(UPLOAD_DIR / uploaded_file.name)
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
                result = run_single_analysis(pdf_path, jd_text, st.session_state["DEEPSEEK_API_KEY"])
                _display_result(result)
            except Exception as e:
                logger.exception("Analysis failed for %s", pdf_path)
                st.error(f"分析过程出错: {str(e)}")
            finally:
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)


def _display_result(result):
    display_analysis_result(result)
