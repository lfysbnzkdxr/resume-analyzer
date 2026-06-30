"""ChromaDB vector store wrapper for resume chunks."""

import logging
import uuid
from pathlib import Path
import threading
import time
import numpy as np
import chromadb
from chromadb.config import Settings

from typing import Optional

from src.core.config import CHROMA_DIR

logger = logging.getLogger(__name__)
from src.rag.embeddings import embed_text, embed_texts

COLLECTION_NAME = "resume_chunks"

# BM25 cache (lazy-built, invalidated on write)
_bm25_index = None  # BM25Okapi instance (lazy)
_bm25_docs: Optional[list[dict]] = None  # parallel list: {id, text, metadata}
_bm25_dirty: bool = True  # True when cache needs rebuild

# ChromaDB client singleton (lazy-initialized)
_client = None

# BM25 thread safety
_bm25_lock = threading.Lock()


def _tokenize(text: str) -> list[str]:
    """Chinese-aware tokenization using jieba for BM25."""
    import jieba
    return list(jieba.cut(text))


def _rebuild_bm25_index():
    """Build (or rebuild) the BM25 index from all stored chunks."""
    from rank_bm25 import BM25Okapi
    global _bm25_index, _bm25_docs, _bm25_dirty

    collection = _get_collection()
    all_data = collection.get(include=["documents", "metadatas"])

    docs = []
    if all_data and all_data["ids"]:
        for i in range(len(all_data["ids"])):
            docs.append({
                "id": all_data["ids"][i],
                "text": all_data["documents"][i],
                "metadata": all_data["metadatas"][i] if all_data.get("metadatas") else {},
            })

    if not docs:
        _bm25_index = None
        _bm25_docs = []
        _bm25_dirty = False
        return

    tokenized_corpus = [_tokenize(d["text"]) for d in docs]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_docs = docs
    _bm25_dirty = False


def _bm25_search(query: str, top_k: int) -> list[dict]:
    """Search via BM25 keyword matching.

    Rebuilds the index if the store has changed since last build.
    Thread-safe via _bm25_lock.
    """
    global _bm25_dirty
    with _bm25_lock:
        if _bm25_dirty or _bm25_index is None:
            _rebuild_bm25_index()

        if _bm25_index is None or not _bm25_docs:
            return []

        tokenized_query = _tokenize(query)
        scores = _bm25_index.get_scores(tokenized_query)

        # Snapshot docs inside the lock to avoid TOCTOU race
        docs_snapshot = list(_bm25_docs)

    # Pair scores with docs and sort (use local snapshot after lock release)
    ranked = sorted(
        [(scores[i], docs_snapshot[i]) for i in range(len(docs_snapshot))],
        key=lambda x: -x[0],
    )

    # Normalise scores to [0, 1] and return top_k
    max_score = max((s for s, _ in ranked), default=1)
    results = []
    for score, doc in ranked[:top_k]:
        results.append({
            "id": doc["id"],
            "text": doc["text"],
            "metadata": doc["metadata"],
            "score": score / max_score if max_score > 0 else 0,
        })
    return results


def _get_client():
    """Get or create the persistent ChromaDB client (singleton)."""
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_collection(client=None):
    """Get or create the resume chunks collection."""
    if client is None:
        client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_resume(filename: str, chunks: list[str], rebuild: bool = True) -> int:
    """Add a resume's chunks to the vector store. Returns chunk count.

    Args:
        filename: Resume file name.
        chunks: List of text chunks.
        rebuild: Rebuild BM25 index immediately after adding.
                 Set to False during batch uploads, then call
                 rebuild_bm25_index() once at the end.
    """
    if not chunks:
        return 0

    t0 = time.perf_counter()
    vectors = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"filename": filename, "chunk_index": i, "chunk_count": len(chunks)}
        for i in range(len(chunks))
    ]

    collection = _get_collection()
    collection.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    logger.debug("[TIMING] rag.vector_store.add_resume (%d chunks) embedding+store=%dms", len(chunks), elapsed_ms)

    global _bm25_dirty
    with _bm25_lock:
        if rebuild:
            _rebuild_bm25_index()
        else:
            _bm25_dirty = True

    return len(chunks)


def rebuild_bm25_index():
    """Explicitly rebuild the BM25 index from all stored chunks.

    Use this after a batch of uploads where each add_resume(rebuild=False)
    was used, to rebuild the index once at the end instead of per file.
    """
    _rebuild_bm25_index()


def search_resumes(
    query: str,
    top_k: int = 5,
    mmr: bool = True,
    lambda_mult: float = 0.7,
    hybrid: bool = True,
) -> list[dict]:
    """Search resume chunks with hybrid retrieval (semantic + BM25 keyword).

    Args:
        query: Search query text.
        top_k: Number of results to return.
        mmr: Enable Max Marginal Relevance for diversity.
        lambda_mult: MMR balance (1.0 = pure relevance, 0.0 = pure diversity).
    """
    collection = _get_collection()

    # --- Semantic search ---
    t0 = time.perf_counter()
    query_vector = embed_text(query)
    fetch_k = top_k * 3 if (mmr or hybrid) else top_k
    include = ["documents", "metadatas", "distances"]
    if mmr:
        include.append("embeddings")

    sem_results = collection.query(
        query_embeddings=[query_vector],
        n_results=fetch_k,
        include=include,
    )
    sem_ms = round((time.perf_counter() - t0) * 1000)

    semantic_hits = {}
    if sem_results["ids"] and sem_results["ids"][0]:
        for i in range(len(sem_results["ids"][0])):
            doc_id = sem_results["ids"][0][i]
            semantic_hits[doc_id] = {
                "id": doc_id,
                "text": sem_results["documents"][0][i],
                "metadata": sem_results["metadatas"][0][i],
                "score": 1 - (sem_results["distances"][0][i] if sem_results["distances"] else 0),
                "embedding": sem_results["embeddings"][0][i] if (mmr and sem_results.get("embeddings")) else None,
            }

    # --- BM25 keyword search ---
    t0 = time.perf_counter()
    bm25_results = _bm25_search(query, top_k=fetch_k)
    bm25_ms = round((time.perf_counter() - t0) * 1000)

    logger.debug(
        "[TIMING] rag.vector_store.search_resumes semantic=%dms bm25=%dms total=%dms",
        sem_ms, bm25_ms, sem_ms + bm25_ms,
    )

    bm25_hits = {r["id"]: r for r in bm25_results}

    # --- Fusion: Reciprocal Rank Fusion ---
    all_ids = set(semantic_hits) | set(bm25_hits)

    def _rrf(match_id: str, rank_constant: int = 60) -> float:
        score = 0.0
        for rank, hit_id in enumerate(
            s["id"] for s in sorted(semantic_hits.values(), key=lambda x: -x["score"])
        ):
            if hit_id == match_id:
                score += 1.0 / (rank_constant + rank)
                break
        if hybrid:
            for rank, hit_id in enumerate(
                b["id"] for b in sorted(bm25_hits.values(), key=lambda x: -x["score"])
            ):
                if hit_id == match_id:
                    score += 1.0 / (rank_constant + rank)
                    break
        return score

    merged = []
    for doc_id in all_ids:
        entry = semantic_hits.get(doc_id) or bm25_hits[doc_id]
        entry["semantic_score"] = entry.get("score", 0)  # 保留语义/Bm25原始分
        entry["score"] = _rrf(doc_id)                     # RRF 仅用于排序
        merged.append(entry)

    merged.sort(key=lambda x: -x["score"])

    # --- MMR diversification ---
    if mmr and len(merged) > top_k:
        merged = _mmr_select(merged, query_vector, top_k, lambda_mult)

    hits = merged[:top_k]

    # Strip embeddings from output
    for h in hits:
        h.pop("embedding", None)

    return hits


def search_resumes_aggregated(
    query: str,
    top_k: int = 5,
    fetch_multiplier: int = 10,
    **kwargs,
) -> list[dict]:
    """Search and aggregate results to resume level.

    Returns top_k unique resumes (by filename), each with:
      - filename: resume file name
      - score: max chunk score for this resume
      - best_chunk: text of the best-matching chunk
      - matching_chunks: list of all matching chunks for this resume
    """
    fetch_k = top_k * fetch_multiplier
    chunks = search_resumes(query, top_k=fetch_k, **kwargs)

    # Aggregate by filename, keeping max score
    resume_map: dict[str, dict] = {}
    for c in chunks:
        metadata = c.get("metadata") or {}
        filename = metadata.get("filename", "未知")
        if filename not in resume_map:
            resume_map[filename] = {
                "filename": filename,
                "score": c["score"],
                "semantic_score": c.get("semantic_score", 0),
                "best_chunk": c["text"],
                "matching_chunks": [c],
            }
        else:
            resume_map[filename]["matching_chunks"].append(c)
            if c["score"] > resume_map[filename]["score"]:
                resume_map[filename]["score"] = c["score"]
                resume_map[filename]["semantic_score"] = c.get("semantic_score", 0)
                resume_map[filename]["best_chunk"] = c["text"]

    aggregated = sorted(resume_map.values(), key=lambda x: -x["semantic_score"])
    return aggregated[:top_k]


def _mmr_select(
    candidates: list[dict],
    query_embedding: list[float],
    top_k: int,
    lambda_mult: float,
) -> list[dict]:
    """Greedy Max Marginal Relevance selection.

    Iteratively picks candidates that balance query relevance against
    similarity to already-selected results.
    """
    query_vec = np.array(query_embedding)
    q_norm = np.linalg.norm(query_vec)
    if q_norm > 0:
        query_vec /= q_norm

    embeds = np.array([c["embedding"] for c in candidates])
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeds /= norms

    rel = embeds @ query_vec
    sim = embeds @ embeds.T

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        mmr_scores = []
        for i in remaining:
            sim_to_selected = max([sim[i][j] for j in selected], default=0)
            mmr_score = lambda_mult * rel[i] - (1 - lambda_mult) * sim_to_selected
            mmr_scores.append(mmr_score)

        best = remaining[int(np.argmax(mmr_scores))]
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]


def list_resumes() -> list[str]:
    """List all unique filenames in the vector store."""
    collection = _get_collection()
    all_meta = collection.get(include=["metadatas"])
    filenames = set()
    if all_meta and all_meta["metadatas"]:
        for m in all_meta["metadatas"]:
            if m and "filename" in m:
                filenames.add(m["filename"])
    return sorted(filenames)


def delete_resume(filename: str) -> int:
    """Delete all chunks for a given filename. Returns count removed."""
    global _bm25_dirty
    collection = _get_collection()
    result = collection.delete(where={"filename": filename})
    deleted = result.get("deleted", 0) if result else 0
    if deleted:
        with _bm25_lock:
            _bm25_dirty = True
    return deleted


def get_resume_text(filename: str) -> str:
    """Retrieve the full concatenated text of a resume from the vector store."""
    collection = _get_collection()
    results = collection.get(
        where={"filename": filename},
        include=["documents", "metadatas"],
    )
    if not results or not results["ids"]:
        return ""
    # Sort by chunk_index to preserve original order
    chunks = sorted(
        [
            (m.get("chunk_index", i), d)
            for i, (d, m) in enumerate(
                zip(results["documents"], results["metadatas"])
            )
        ],
        key=lambda x: x[0],
    )
    return "\n".join(text for _, text in chunks)


def get_resume_count() -> int:
    """Get the total number of unique resumes in the store."""
    return len(list_resumes())
