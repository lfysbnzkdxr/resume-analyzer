"""ChromaDB vector store wrapper for resume chunks."""

import uuid
from pathlib import Path
import chromadb
from chromadb.config import Settings

from src.core.config import CHROMA_DIR
from src.rag.embeddings import embed_text, embed_texts

COLLECTION_NAME = "resume_chunks"


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


def search_resumes(query: str, top_k: int = 5) -> list[dict]:
    """Search resume chunks by semantic similarity to the query."""
    query_vector = embed_text(query)
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - (results["distances"][0][i] if results["distances"] else 0),
            })
    return hits


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
