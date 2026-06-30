"""Library match page — paste a JD, search for best matching resumes."""

import streamlit as st
from src.rag.retriever import retrieve_resumes_aggregated
from src.rag.vector_store import get_resume_text
from src.core.orchestrator import run_text_analysis
from src.ui.components.score_chart import display_analysis_result
from src.ui.theme import score_color


def render():
    st.header("🔍 库匹配检索")
    st.markdown("粘贴职位描述，从简历库中检索最匹配的候选人。")

    jd_text = st.text_area(
        "粘贴职位描述 (JD)",
        height=250,
        placeholder="我们正在寻找一位 AI 工程师...",
    )

    # --- Search ---
    if st.button("🔍 从库中检索最佳匹配", type="primary", use_container_width=True):
        if not jd_text.strip():
            st.error("请输入职位描述")
            return

        with st.spinner("正在检索..."):
            results = retrieve_resumes_aggregated(jd_text, top_k=5)

        st.session_state["match_results"] = results
        st.session_state.pop("match_analysis", None)
        st.rerun()

    # --- Display cached results ---
    results = st.session_state.get("match_results")
    if not results:
        return

    st.subheader(f"前 {len(results)} 名匹配候选人")

    for rank, r in enumerate(results, 1):
        filename = r["filename"]
        score = r.get("semantic_score", r["score"])
        pct = max(0, min(int(score * 100), 100))
        chunk_count = len(r["matching_chunks"])

        color = score_color(pct)
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

        if st.button(f"📄 查看详细分析 - {filename}", key=f"analyze_{filename}"):
            with st.spinner("AI 分析中...（简历提取 → JD 分析 → 匹配评估）"):
                try:
                    resume_text = get_resume_text(filename)
                    if not resume_text:
                        st.error(f"无法从库中读取简历: {filename}")
                        st.stop()

                    result = run_text_analysis(resume_text, filename, jd_text, st.session_state["DEEPSEEK_API_KEY"])
                    st.session_state["match_analysis"] = {
                        "filename": filename,
                        "result": result,
                    }
                except Exception as e:
                    st.error(f"分析过程出错: {str(e)}")
            st.rerun()

    # --- Display detailed analysis ---
    analysis = st.session_state.get("match_analysis")
    if analysis:
        st.markdown("---")
        st.markdown(f"## 📄 {analysis['filename']} 详细分析")
        display_analysis_result(analysis["result"])

        if st.button("✕ 关闭详细分析", key="close_analysis"):
            st.session_state.pop("match_analysis", None)
            st.rerun()
