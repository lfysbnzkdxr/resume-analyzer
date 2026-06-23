"""Library match page — paste a JD, search for best matching resumes."""

import streamlit as st
from src.rag.retriever import retrieve_resumes_aggregated


def render():
    st.header("🔍 库匹配检索")
    st.markdown("粘贴职位描述，从简历库中检索最匹配的候选人。")

    jd_text = st.text_area(
        "粘贴职位描述 (JD)",
        height=250,
        placeholder="我们正在寻找一位 AI 工程师...",
    )

    if st.button("🔍 从库中检索最佳匹配", type="primary", use_container_width=True):
        if not jd_text.strip():
            st.error("请输入职位描述")
            return

        with st.spinner("正在检索..."):
            results = retrieve_resumes_aggregated(jd_text, top_k=5)

        if not results:
            st.warning("未找到匹配的简历。请先上传简历到简历库。")
            return

        st.subheader(f"前 {len(results)} 名匹配候选人")

        for rank, r in enumerate(results, 1):
            filename = r["filename"]
            score = r.get("semantic_score", r["score"])
            pct = max(0, min(int(score * 100), 100))
            chunk_count = len(r["matching_chunks"])

            color = "#22c55e" if pct >= 80 else "#eab308" if pct >= 60 else "#ef4444"
            st.markdown(
                f"""<div style="padding:12px;margin:8px 0;border:1px solid #e5e7eb;border-radius:8px;">
                    <div style="display:flex;justify-content:space-between;">
                        <strong>#{rank} {filename}</strong>
                        <span style="color:{color};font-weight:bold;">{pct}%</span>
                    </div>
                    <div style="color:#888;font-size:12px;margin:4px 0;">
                        匹配片段数: {chunk_count}
                    </div>
                    <div style="background:#e5e7eb;border-radius:4px;height:8px;margin:8px 0;">
                        <div style="background:{color};width:{pct}%;height:8px;border-radius:4px;"></div>
                    </div>
                    <details>
                        <summary>查看最佳匹配片段</summary>
                        <pre style="font-size:13px;margin-top:8px;">{r['best_chunk'][:500]}</pre>
                    </details>
                </div>""",
                unsafe_allow_html=True,
            )
