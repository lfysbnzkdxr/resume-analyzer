"""Streamlit main application — multi-page app."""

import streamlit as st
import os
from src.core.config import APP_TITLE, APP_VERSION, DEEPSEEK_API_KEY

PAGES = {
    "single_analysis": "📄 单份分析",
    "library_manage": "📚 简历库",
    "library_match": "🔍 库匹配",
}


def _inject_css():
    """注入 CSS 隐藏 Streamlit 内置英文 UI 元素。"""
    st.markdown("""
<style>
/* 隐藏 Streamlit 底部信息 */
footer { display: none !important; }

/* 文件上传器 — 替换 "Drag and drop" 为中文 */
[data-testid="stFileUploaderDropzone"] div:first-child { display: none !important; }
[data-testid="stFileUploaderDropzone"]::before {
    content: "拖拽文件到此处，或点击选择文件";
    display: block;
    padding: 10px;
    text-align: center;
    color: #666;
}

/* 文件上传器 — 替换 "Browse files" 为中文 */
[data-testid="stFileUploaderDropzone"] button { font-size: 0; }
[data-testid="stFileUploaderDropzone"] button::before {
    content: "选择文件";
    font-size: 14px;
}

/* 文件上传器 — 替换 "Limit 200MB per file" 为中文 */
[data-testid="stFileUploaderDropzone"] small { font-size: 0; }
[data-testid="stFileUploaderDropzone"] small::before {
    content: "单个文件限制 200MB";
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _inject_css()

    with st.sidebar:
        st.title(APP_TITLE)
        st.divider()

        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=DEEPSEEK_API_KEY,
            help="输入你的 DeepSeek API Key，不会保存到代码中",
        )
        if api_key:
            st.session_state["DEEPSEEK_API_KEY"] = api_key

        st.divider()

        page = st.radio("导航", options=list(PAGES.keys()), format_func=lambda k: PAGES[k])

        st.divider()
        st.caption(f"AI 简历智能分析 v{APP_VERSION}")
        st.caption("基于 DeepSeek + RAG + 多 Agent 协作")

    if not api_key and not DEEPSEEK_API_KEY:
        st.warning("请在侧边栏输入 DeepSeek API Key 以开始使用")
        return

    if page == "single_analysis":
        from src.ui.pages.single_analysis import render as r
    elif page == "library_manage":
        from src.ui.pages.library_manage import render as r
    elif page == "library_match":
        from src.ui.pages.library_match import render as r

    r()


if __name__ == "__main__":
    main()
