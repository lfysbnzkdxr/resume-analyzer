"""Library management page — upload/list/delete resumes in the vector store."""

import streamlit as st
from pathlib import Path

from src.agents.library_agent import (
    add_resume_to_library,
    rebuild_library_index,
    list_library_resumes,
    remove_resume_from_library,
    get_library_stats,
)


def render():
    st.header("📚 简历库管理")
    st.markdown("上传多份简历到本地向量库，支持语义检索匹配。")

    stats = get_library_stats()
    st.info(f"当前库中有 **{stats['resume_count']}** 份简历")

    uploaded_files = st.file_uploader(
        "上传简历 (PDF，可多选)",
        type=["pdf"],
        accept_multiple_files=True,
        key="lib_upload",
    )

    if uploaded_files:
        tmp_dir = Path("data/uploads")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        progress = st.progress(0, text="处理中...")

        for i, f in enumerate(uploaded_files):
            pdf_path = str(tmp_dir / f.name)
            with open(pdf_path, "wb") as fp:
                fp.write(f.getbuffer())

            with st.spinner(f"处理: {f.name}"):
                result = add_resume_to_library(pdf_path, rebuild=False)

            if result["success"]:
                st.success(f"✅ {f.name} — 已入库 ({result['chunks']} 个文本块)")
            else:
                st.error(f"❌ {f.name} — {result.get('error', '处理失败')}")

            Path(pdf_path).unlink(missing_ok=True)
            progress.progress((i + 1) / len(uploaded_files))

        # Build BM25 index once after all files are uploaded
        with st.spinner("正在构建检索索引..."):
            rebuild_library_index()
        progress.empty()
        st.rerun()

    st.divider()
    st.subheader("已入库简历")

    resumes = list_library_resumes()
    if not resumes:
        st.info("暂无简历，请上传")
    else:
        for r in resumes:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {r}")
            with col2:
                if st.button("删除", key=f"del_{r}"):
                    removed = remove_resume_from_library(r)
                    if removed > 0:
                        st.success(f"已删除: {r}")
                        st.rerun()
                    else:
                        st.error("删除失败")
