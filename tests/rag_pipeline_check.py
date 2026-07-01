"""RAG Pipeline end-to-end verification test.

Tests the full pipeline:
  1. Create test PDF → 2. Upload to library → 3. Semantic search → 4. Verify → 5. Cleanup
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_pdf_creation():
    """Step 1: Create a test PDF with known content."""
    from src.tools.pdf_parser import parse_resume_pdf

    content = """个人信息：李四 | li.si@email.com | 139-0000-0002

个人总结：2年后端开发经验，主要在电商领域，熟悉Java生态，对AI有学习意愿。

技能：Java, Spring Boot, MySQL, Redis, Docker, Kafka, Python（基础）

工作经验：
公司：美团 (2022-2024) | 职位：Java后端开发
- 负责订单系统的设计和开发
- 使用Spring Boot构建微服务
- 使用Redis做缓存优化，QPS提升40%
- 维护Kafka消息队列

项目经验：
项目：电商订单管理系统
- 基于Spring Cloud的微服务架构
- 使用Docker进行容器化部署

教育背景：
学校：华中科技大学 | 学位：本科 | 专业：软件工程
"""

    pdf_path = PROJECT_ROOT / "data" / "test_rag_李四.pdf"
    # Create a valid PDF using PyMuPDF (fitz) directly
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), content, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()
    print(f"✅ PDF created: {pdf_path} ({pdf_path.stat().st_size} bytes)")

    # Verify PyMuPDF can parse it
    text = parse_resume_pdf(str(pdf_path))
    assert text and len(text.strip()) > 50, f"PyMuPDF parsed text too short: {len(text.strip()) if text else 0}"
    print(f"✅ PyMuPDF parsed OK: {len(text.strip())} chars extracted")
    return str(pdf_path)


def test_upload_to_library(pdf_path: str):
    """Step 2: Upload the test PDF to the ChromaDB library."""
    from src.rag.library import add_resume_to_library, get_library_stats

    result = add_resume_to_library(pdf_path)
    assert result["success"], f"Upload failed: {result.get('error')}"
    assert result["chunks"] > 0, f"No chunks created: {result}"
    print(f"✅ Uploaded to library: {result['chunks']} chunks, {result['text_length']} chars")

    stats = get_library_stats()
    assert stats["resume_count"] >= 1, f"Library stats unexpected: {stats}"
    print(f"✅ Library stats: {stats['resume_count']} resumes")

    return result


def test_semantic_search():
    """Step 3-4: Search the library and verify relevance."""
    from src.rag.vector_store import search_resumes

    # Search for something related to the uploaded resume
    results = search_resumes("Java Spring Boot 后端开发", top_k=3)
    assert len(results) > 0, "Search returned no results!"
    print(f"✅ Search returned {len(results)} results")

    # Check result structure
    r = results[0]
    assert "text" in r, f"Result missing 'text': {r.keys()}"
    assert "metadata" in r, f"Result missing 'metadata': {r.keys()}"
    assert "score" in r, f"Result missing 'score': {r.keys()}"
    assert r["score"] > 0.02, f"Top result RRF score too low: {r['score']:.4f}"
    print(f"[OK] Top result: score={r['score']:.4f}, file={r['metadata'].get('filename')}")

    # Verify the top result is our test resume
    assert "test_rag_李四" in r["metadata"].get("filename", ""), (
        f"Top result is not our test resume: {r['metadata'].get('filename')}"
    )
    print("✅ Top result is our test resume — relevance verified")

    # Also test retrieval via high-level function
    from src.rag.retriever import retrieve_resumes_for_jd

    jd_results = retrieve_resumes_for_jd("我们需要Java后端开发，熟悉Spring Boot、MySQL", top_k=3)
    assert len(jd_results) > 0, "JD-based retrieval returned no results!"
    jd_score = jd_results[0]["score"]
    assert jd_score > 0.01, f"JD-based retrieval RRF score too low: {jd_score:.4f}"
    print(f"[OK] JD-based retrieval: top score={jd_score:.4f}")

    return results


def test_cleanup(pdf_path: str):
    """Step 5: Clean up test data."""
    from src.rag.library import remove_resume_from_library

    filename = Path(pdf_path).name
    removed = remove_resume_from_library(filename)
    assert removed > 0, f"Failed to delete resume '{filename}'"
    print(f"✅ Cleaned up: removed {removed} chunks for '{filename}'")

    Path(pdf_path).unlink(missing_ok=True)
    print(f"✅ Cleaned up: deleted '{pdf_path}'")


def main():
    print("=" * 60)
    print("RAG Pipeline End-to-End Verification")
    print("=" * 60)

    pdf_path = None
    failures = []

    try:
        # Step 1
        print("\n[Step 1/5] Creating test PDF...")
        pdf_path = test_pdf_creation()
    except Exception as e:
        failures.append(f"Step 1 (PDF creation): {e}")

    try:
        # Step 2
        print("\n[Step 2/5] Uploading to library...")
        test_upload_to_library(pdf_path)
    except Exception as e:
        failures.append(f"Step 2 (Upload): {e}")

    try:
        # Step 3-4
        print("\n[Step 3-4/5] Searching and verifying...")
        test_semantic_search()
    except Exception as e:
        failures.append(f"Step 3-4 (Search): {e}")

    try:
        # Step 5
        print("\n[Step 5/5] Cleaning up...")
        if pdf_path and Path(pdf_path).exists():
            test_cleanup(pdf_path)
        else:
            print("⚠️  Nothing to clean up")
    except Exception as e:
        failures.append(f"Step 5 (Cleanup): {e}")

    print("\n" + "=" * 60)
    if failures:
        print(f"❌ VERIFICATION FAILED — {len(failures)} error(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("✅ RAG PIPELINE VERIFIED — all steps passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
