"""ChromaDB vector store wrapper for resume chunks."""

import uuid
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings

from typing import Optional

from src.core.config import CHROMA_DIR
from src.rag.embeddings import embed_text, embed_texts

COLLECTION_NAME = "resume_chunks"

# BM25 cache (lazy-built, invalidated on write)
_bm25_index = None  # BM25Okapi instance (lazy)
_bm25_docs: Optional[list[dict]] = None  # parallel list: {id, text, metadata}
_bm25_doc_count: int = 0


def _tokenize(text: str) -> list[str]:
    """Chinese-aware tokenization using jieba for BM25."""
    import jieba
    return list(jieba.cut(text))


def _rebuild_bm25_index():
    """Build (or rebuild) the BM25 index from all stored chunks."""
    from rank_bm25 import BM25Okapi
    global _bm25_index, _bm25_docs, _bm25_doc_count

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
        _bm25_doc_count = 0
        return

    tokenized_corpus = [_tokenize(d["text"]) for d in docs]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_docs = docs
    _bm25_doc_count = len(docs)


def _bm25_search(query: str, top_k: int) -> list[dict]:
    """Search via BM25 keyword matching.

    Rebuilds the index if the store has changed since last build.
    """
    collection = _get_collection()
    current_count = collection.count()
    global _bm25_doc_count
    if _bm25_index is None or current_count != _bm25_doc_count:
        _rebuild_bm25_index()

    if _bm25_index is None or not _bm25_docs:
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    # Pair scores with docs and sort
    ranked = sorted(
        [(scores[i], _bm25_docs[i]) for i in range(len(_bm25_docs))],
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
    """Get or create the persistent ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(client=None):
    """Get or create the resume chunks collection."""
    if client is None:
        client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_resume(filename: str, chunks: list[str]) -> int:
    """Add a resume's chunks to the vector store. Returns chunk count."""
    if not chunks:
        return 0

    vectors = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"filename": filename, "chunk_index": i, "chunk_count": len(chunks)}
        for i in range(len(chunks))
    ]

    collection = _get_collection()
    collection.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)
    return len(chunks)


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
    bm25_results = _bm25_search(query, top_k=fetch_k)
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
        entry["score"] = _rrf(doc_id)
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
    import numpy as np

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
    collection = _get_collection()
    all_data = collection.get(include=["metadatas"])
    to_delete = []
    if all_data and all_data["ids"] and all_data["metadatas"]:
        for i, m in enumerate(all_data["metadatas"]):
            if m and m.get("filename") == filename:
                to_delete.append(all_data["ids"][i])
    if to_delete:
        collection.delete(ids=to_delete)
    return len(to_delete)


def get_resume_count() -> int:
    """Get the total number of unique resumes in the store."""
    return len(list_resumes())
