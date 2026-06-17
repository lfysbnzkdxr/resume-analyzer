"""RAG retrieval quality benchmark.

Compares:
  1. Section-aware vs naive fixed-size chunking  (Hit Rate, MRR)
  2. Hybrid (semantic+BM25) vs semantic-only search

Usage:
    python eval/rag_benchmark.py

Note: Results are meaningful as a REGRESSION guard — if a change to splitter,
embedder, or search logic drops these numbers, something broke.
For absolute differentiation between strategies, run with a larger
corpus of realistic resumes (20+ documents).
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embeddings import embed_text, embed_texts

# ---------------------------------------------------------------------------
# Test corpus — multi-section resumes covering AI/Java/Frontend/RecSys
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "id": "ai_algo",
        "filename": "bench_张三_AI算法.pdf",
        "text": """
个人信息：张三 | zhang.san@email.com | 博士在读

个人总结：3年AI算法经验，主要研究和应用方向为自然语言处理和大语言模型，有RAG系统落地经验。

技能：Python, PyTorch, TensorFlow, LangChain, ChromaDB, Transformers, vLLM, RAG, BERT, DeepSpeed, FastAPI, Docker, Git

工作经验：
公司：字节跳动 (2022-2025) | 职位：AI算法工程师
- 基于LangChain + ChromaDB构建RAG pipeline，召回率达92%
- 使用vLLM部署大模型推理服务，TP99延迟控制在200ms以内
- 使用DeepSpeed进行大模型分布式训练，64卡训练效率达到92%
- 基于FastAPI构建模型推理服务，日均处理10万+请求

项目经验：
项目一：企业智能知识库问答系统
- 基于LangChain + ChromaDB + DeepSeek API搭建RAG系统
- 使用fastembed做本地向量化，离线部署无需GPU
- 实现多维度检索（语义+关键词）确保召回质量
项目二：文本分类微调平台
- 基于BERT的文本分类模型微调训练
- 支持多标签分类和少样本学习

教育背景：
学校：北京大学 | 学位：硕士 | 专业：计算机科学与技术
""",
    },
    {
        "id": "java_backend",
        "filename": "bench_李四_Java后端.pdf",
        "text": """
个人信息：李四 | li.si@email.com | 本科

个人总结：3年Java后端开发经验，熟悉微服务架构和常用中间件，有高并发系统设计经验。

技能：Java, Spring Boot, Spring Cloud, MyBatis, MySQL, Redis, Docker, Kafka, Elasticsearch, Nginx, Linux

工作经验：
公司：美团 (2022-2025) | 职位：Java后端开发
- 基于Spring Cloud构建订单微服务系统，日均处理百万级订单
- 使用Redis缓存优化热点数据查询，QPS提升40%
- 维护Kafka消息队列，处理日均千万级异步消息
- 搭建ELK日志平台，支持分布式系统链路追踪

项目经验：
项目一：电商订单管理系统
- 基于Spring Cloud的微服务架构设计
- 使用Redis + 本地缓存构建多级缓存
- 基于Kafka实现订单状态的异步通知
项目二：分布式日志采集平台
- 基于ELK Stack构建分布式日志采集
- 日志量日均50GB，支持实时搜索和聚合分析

教育背景：
学校：华中科技大学 | 学位：本科 | 专业：软件工程
""",
    },
    {
        "id": "java_architect",
        "filename": "bench_钱七_Java架构.pdf",
        "text": """
个人信息：钱七 | qian.qi@email.com | 硕士

个人总结：6年Java开发经验，3年架构设计经验，主导过亿级流量电商系统架构升级，有技术团队管理经验。

技能：Java, Spring Cloud, Spring Boot, Dubbo, ZooKeeper, Nacos, Sentinel, Redis, MySQL, RocketMQ, Nginx, Docker, Kubernetes, MyBatis, ELK, XXL-Job

工作经验：
公司：京东 (2019-2025) | 职位：技术专家
- 主导电商交易核心系统从单体架构拆分为微服务架构
- 自研分布式事务中间件TXC，支撑日均千万级订单
- 基于Nacos和Spring Cloud Gateway搭建微服务网关层
- 使用Sentinel实现流量控制和熔断降级
- 构建ELK日志平台，日均处理50TB日志数据

项目经验：
项目一：分布式事务中间件TXC
- 基于RocketMQ实现最终一致性事务消息
- 支持TCC和Saga两种事务模式
- 接入100+核心服务，日均处理500万+事务请求
项目二：全链路压测平台
- 模拟真实流量，支持百万并发压测
- 自动采集压测数据和性能瓶颈分析

教育背景：
学校：浙江大学 | 学位：硕士 | 专业：计算机技术
""",
    },
    {
        "id": "frontend",
        "filename": "bench_王五_前端.pdf",
        "text": """
个人信息：王五 | wang.wu@email.com | 本科

个人总结：5年前端开发经验，擅长React生态和大型前端工程化架构，有微前端落地经验。

技能：JavaScript, TypeScript, React, Vue, Next.js, Webpack, Vite, ECharts, D3.js, Node.js, Jest, Cypress, GraphQL, WebSocket

工作经验：
公司：阿里巴巴 (2020-2025) | 职位：高级前端工程师
- 开发数据大屏可视化平台，基于ECharts + React + D3.js
- 搭建前端组件库，支撑20+业务线统一UI
- 推动微前端架构落地，解决巨石应用拆分问题

项目经验：
项目一：数据可视化大屏平台
- 基于ECharts和D3.js实现20+可视化图表组件
- WebSocket推送实时数据更新
项目二：前端监控系统
- 基于Performance API实现性能监控
- 错误采集覆盖JS异常和资源加载失败

教育背景：
学校：浙江大学 | 学位：本科 | 专业：信息工程
""",
    },
    {
        "id": "recommender",
        "filename": "bench_孙八_搜推算法.pdf",
        "text": """
个人信息：孙八 | sun.ba@email.com | 硕士

个人总结：3年搜推算法经验，熟悉推荐系统召回排序全流程和实时特征工程，有大规模推荐系统落地经验。

技能：Python, TensorFlow, PyTorch, Flink, Hadoop, Hive, Spark, Kafka, ClickHouse, A/B测试, DolphinScheduler, Redis

工作经验：
公司：百度 (2022-2025) | 职位：推荐算法工程师
- 设计并上线多路召回融合策略，首页CTR提升12%
- 构建实时特征平台，基于Flink + Kafka处理用户实时行为数据
- 基于PyTorch训练深度兴趣网络DIN模型
- 搭建A/B测试平台，支撑每日50+实验

项目经验：
项目一：短视频推荐系统
- 多路召回策略（协同过滤 + 向量召回 + 热度召回）
- 基于DIN的点击率预估模型
项目二：实时特征计算平台
- 基于Flink + Kafka实现毫秒级特征计算
- 特征数据写入ClickHouse支持即席分析

教育背景：
学校：北京航空航天大学 | 学位：硕士 | 专业：计算机科学与技术
""",
    },
]

# ---------------------------------------------------------------------------
# Queries — each targets exactly one doc
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "query": "RAG 问答系统 LangChain ChromaDB 检索增强生成 DeepSeek",
        "target_doc": "ai_algo",
        "note": "AI — RAG / LangChain 经验",
    },
    {
        "query": "Java Spring Boot MyBatis MySQL 订单系统 微服务",
        "target_doc": "java_backend",
        "note": "Java后端 — Spring Boot / 微服务",
    },
    {
        "query": "分布式事务 RocketMQ Nacos Sentinel 微服务网关 架构设计",
        "target_doc": "java_architect",
        "note": "Java架构 — 分布式事务 / 服务治理",
    },
    {
        "query": "React ECharts D3.js 数据可视化 大屏 WebSocket",
        "target_doc": "frontend",
        "note": "前端 — React / ECharts / 可视化",
    },
    {
        "query": "推荐系统 CTR预估 DIN 多路召回 排序模型 协同过滤",
        "target_doc": "recommender",
        "note": "搜推 — 推荐系统 / CTR / 召回",
    },
    {
        "query": "DeepSpeed vLLM 大模型推理 分布式训练 模型部署",
        "target_doc": "ai_algo",
        "note": "AI — 大模型推理/训练精确技能匹配",
    },
    {
        "query": "Kafka Redis 缓存 高并发 消息队列 Elasticsearch",
        "target_doc": "java_backend",
        "note": "后端 — 中间件/高并发精确匹配",
    },
    {
        "query": "Docker Kubernetes Nacos Dubbo ZooKeeper 容器化 服务发现",
        "target_doc": "java_architect",
        "note": "架构 — K8s/服务发现精确匹配",
    },
    {
        "query": "Flink ClickHouse DolphinScheduler Hive 实时特征 大数据",
        "target_doc": "recommender",
        "note": "搜推 — Flink/ClickHouse 精确匹配",
    },
]


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------
def _naive_split(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    text = text.strip()
    if not text:
        return []
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap if end < len(text) else end
    return chunks


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def hit_rate(results: list[list[dict]], target_ids: list[str]) -> float:
    hits = 0
    for res, tid in zip(results, target_ids):
        for r in res:
            if r["metadata"].get("doc_id") == tid:
                hits += 1
                break
    return hits / len(results) if results else 0.0


def mrr(results: list[list[dict]], target_ids: list[str]) -> float:
    total = 0.0
    for res, tid in zip(results, target_ids):
        for rank, r in enumerate(res, start=1):
            if r["metadata"].get("doc_id") == tid:
                total += 1.0 / rank
                break
    return total / len(results) if results else 0.0


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_benchmark(top_k: int = 3):
    """Run both comparisons, print report, return metrics dict."""
    import chromadb
    from chromadb.config import Settings

    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="rag_bench_"))

    client = chromadb.PersistentClient(
        path=str(tmpdir),
        settings=Settings(anonymized_telemetry=False),
    )

    from src.rag.splitter import split_text as section_split

    def _get_collection(name):
        try:
            client.delete_collection(name)
        except Exception:
            pass
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    print("=" * 65)
    print("RAG Retrieval Quality Benchmark")
    print(f"top_k={top_k}  |  docs={len(DOCUMENTS)}  |  queries={len(QUERIES)}")
    print("=" * 65)

    # ---- Index: section-aware split ----
    print("\n[1] Section-aware split:")
    coll_section = _get_collection("section")
    for doc in DOCUMENTS:
        chunks = section_split(doc["text"], chunk_size=500, chunk_overlap=50)
        ids = [str(uuid.uuid4()) for _ in chunks]
        vectors = embed_texts(chunks)
        metas = [{"doc_id": doc["id"], "filename": doc["filename"]} for _ in chunks]
        coll_section.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metas)
        print(f"  ✓ {doc['filename']}: {len(chunks)} chunks")

    # ---- Index: naive split ----
    print("\n[2] Naive fixed-size split:")
    coll_naive = _get_collection("naive")
    for doc in DOCUMENTS:
        chunks = _naive_split(doc["text"], chunk_size=500, chunk_overlap=50)
        ids = [str(uuid.uuid4()) for _ in chunks]
        vectors = embed_texts(chunks)
        metas = [{"doc_id": doc["id"], "filename": doc["filename"]} for _ in chunks]
        coll_naive.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metas)
        print(f"  ✓ {doc['filename']}: {len(chunks)} chunks")

    # ---- Search helper ----
    def _search(collection, qvec, k):
        res = collection.query(
            query_embeddings=[qvec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        if res["ids"] and res["ids"][0]:
            for i in range(len(res["ids"][0])):
                hits.append({
                    "id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "score": 1 - (res["distances"][0][i] if res["distances"] else 0),
                })
        return hits

    queries = [q["query"] for q in QUERIES]
    targets = [q["target_doc"] for q in QUERIES]
    query_vectors = [embed_text(q) for q in queries]

    # ---- Comparison 1: Section-aware vs Naive split ----
    print(f"\n{'=' * 65}")
    print(f"Comparison 1: Section-aware split vs Naive split  (top-{top_k})")
    print(f"{'=' * 65}")

    results_section = [_search(coll_section, qv, top_k) for qv in query_vectors]
    results_naive = [_search(coll_naive, qv, top_k) for qv in query_vectors]

    hr_section = hit_rate(results_section, targets)
    hr_naive = hit_rate(results_naive, targets)
    mrr_section = mrr(results_section, targets)
    mrr_naive = mrr(results_naive, targets)

    print(f"{'Metric':<30} {'Section':>10} {'Naive':>10} {'Δ':>10}")
    print("-" * 65)
    for label, sv, nv in [
        ("Hit Rate", hr_section, hr_naive),
        ("MRR", mrr_section, mrr_naive),
    ]:
        d = sv - nv
        ds = f"+{d:.1%}" if d > 0 else f"{d:.1%}"
        print(f"{label:<30} {sv:>10.1%} {nv:>10.1%} {ds:>10}")

    print(f"\n{'─' * 65}")
    print("Per-query (section):")
    for q, res, t in zip(queries, results_section, targets):
        found = next((i + 1 for i, r in enumerate(res) if r["metadata"].get("doc_id") == t), None)
        s = f"OK rank {found}" if found else "MISS"
        print(f"  [{s}] {q[:55]}...")

    print(f"\n{'─' * 65}")
    print("Per-query (naive):")
    for q, res, t in zip(queries, results_naive, targets):
        found = next((i + 1 for i, r in enumerate(res) if r["metadata"].get("doc_id") == t), None)
        s = f"OK rank {found}" if found else "MISS"
        print(f"  [{s}] {q[:55]}...")

    # ---- Comparison 2: Hybrid vs Semantic-only ----
    from src.rag.vector_store import _tokenize

    print(f"\n{'=' * 65}")
    print(f"Comparison 2: Hybrid (semantic+BM25) vs Semantic-only  (top-{top_k})")
    print(f"{'=' * 65}")

    sem_results = results_section

    all_chunks = coll_section.get(include=["documents", "metadatas"])
    from rank_bm25 import BM25Okapi
    bm25_docs = []
    tokenized_corpus = []
    if all_chunks["ids"]:
        for i in range(len(all_chunks["ids"])):
            text = all_chunks["documents"][i]
            tokenized_corpus.append(_tokenize(text))
            bm25_docs.append({
                "id": all_chunks["ids"][i],
                "text": text,
                "metadata": all_chunks["metadatas"][i],
            })
        bm25_index = BM25Okapi(tokenized_corpus)

    def _hybrid_search(q_text, qvec, k):
        sem = _search(coll_section, qvec, k * 4)
        sem_map = {s["id"]: s for s in sem}
        tokenized = _tokenize(q_text)
        bm25_scores = bm25_index.get_scores(tokenized) if bm25_index else []
        ranked = sorted(
            [(bm25_scores[i], bm25_docs[i]) for i in range(len(bm25_docs))],
            key=lambda x: -x[0],
        )
        max_s = max((s for s, _ in ranked), default=1)
        bm25_hits = {}
        for sc, doc in ranked[:k * 4]:
            bm25_hits[doc["id"]] = {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": sc / max_s if max_s > 0 else 0,
            }

        def rrf(mid, rc=60):
            s = 0.0
            ra = sorted(sem_map.values(), key=lambda x: -x["score"])
            for rk, sid in enumerate(x["id"] for x in ra):
                if sid == mid:
                    s += 1.0 / (rc + rk)
                    break
            rb = sorted(bm25_hits.values(), key=lambda x: -x["score"])
            for rk, bid in enumerate(x["id"] for x in rb):
                if bid == mid:
                    s += 1.0 / (rc + rk)
                    break
            return s

        merged = []
        for did in set(sem_map) | set(bm25_hits):
            entry = sem_map.get(did) or bm25_hits[did]
            entry["score"] = rrf(did)
            merged.append(entry)
        merged.sort(key=lambda x: -x["score"])
        return merged[:k]

    hybrid_results = [_hybrid_search(qt, qv, top_k) for qt, qv in zip(queries, query_vectors)]

    hr_hyb = hit_rate(hybrid_results, targets)
    hr_sem_val = hit_rate(sem_results, targets)
    mrr_hyb = mrr(hybrid_results, targets)
    mrr_sem_val = mrr(sem_results, targets)

    print(f"{'Metric':<30} {'Hybrid':>10} {'Semantic':>10} {'Δ':>10}")
    print("-" * 65)
    for label, hv, sv in [
        ("Hit Rate", hr_hyb, hr_sem_val),
        ("MRR", mrr_hyb, mrr_sem_val),
    ]:
        d = hv - sv
        ds = f"+{d:.1%}" if d > 0 else f"{d:.1%}"
        print(f"{label:<30} {hv:>10.1%} {sv:>10.1%} {ds:>10}")

    print(f"\n{'─' * 65}")
    print("Per-query detail (hybrid):")
    for q, res_h, res_s, t in zip(queries, hybrid_results, sem_results, targets):
        h_f = next((i + 1 for i, r in enumerate(res_h) if r["metadata"].get("doc_id") == t), None)
        s_f = next((i + 1 for i, r in enumerate(res_s) if r["metadata"].get("doc_id") == t), None)
        s_s = f"OK rank {s_f}" if s_f else "MISS"
        h_s = f"OK rank {h_f}" if h_f else "MISS"
        marker = " <-- hybrid better" if s_f != h_f else ""
        print(f"  [sem:{s_s} hyb:{h_s}]{marker} {q[:50]}...")

    # ---- Summary ----
    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Section-split Hit Rate:   {hr_section:.1%}")
    print(f"  Naive-split Hit Rate:     {hr_naive:.1%}")
    print(f"  Section-split MRR:        {mrr_section:.2%}")
    print(f"  Naive-split MRR:          {mrr_naive:.2%}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Hybrid Hit Rate:          {hr_hyb:.1%}")
    print(f"  Semantic-only Hit Rate:   {hr_sem_val:.1%}")
    print(f"  Hybrid MRR:               {mrr_hyb:.2%}")
    print(f"  Semantic-only MRR:        {mrr_sem_val:.2%}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Queries: {len(QUERIES)}")
    print(f"  This is a SMALL corpus (5 docs). Numbers are meaningful")
    print(f"  as REGRESSION guards against broken changes.")
    print(f"{'=' * 65}")

    # Cleanup — close client, then remove dir
    del client
    import shutil
    shutil.rmtree(str(tmpdir), ignore_errors=True)

    return {
        "section_split": {"hit_rate": hr_section, "mrr": mrr_section},
        "naive_split": {"hit_rate": hr_naive, "mrr": mrr_naive},
        "hybrid": {"hit_rate": hr_hyb, "mrr": mrr_hyb},
        "semantic_only": {"hit_rate": hr_sem_val, "mrr": mrr_sem_val},
    }


if __name__ == "__main__":
    run_benchmark()
