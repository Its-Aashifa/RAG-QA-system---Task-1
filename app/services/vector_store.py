"""
Embedding generation and FAISS vector store management.

Embedding model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - Fast inference, good semantic quality
  - Runs locally — no API key needed
  - ~80MB download on first use

FAISS index type: IndexFlatIP (Inner Product / cosine similarity)
  - Exact search (no approximation) — suitable for datasets up to ~100k chunks
  - Vectors are L2-normalized before insertion, so inner product == cosine similarity
  - For larger corpora, switch to IndexIVFFlat with nlist=100
"""

import os
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded globals to avoid import-time overhead
_embedding_model = None
_faiss_index = None
_metadata_store: list[dict] = []  # parallel list to FAISS vectors


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            from app.core.config import settings
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    return _embedding_model


def _get_index_paths():
    from app.core.config import settings
    base = settings.FAISS_INDEX_PATH
    return f"{base}.index", f"{base}.meta"


def _load_index():
    global _faiss_index, _metadata_store
    index_path, meta_path = _get_index_paths()

    if os.path.exists(index_path) and os.path.exists(meta_path):
        try:
            import faiss
            _faiss_index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                _metadata_store = pickle.load(f)
            logger.info(f"Loaded FAISS index with {_faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            _faiss_index = None
            _metadata_store = []


def _save_index():
    import faiss
    index_path, meta_path = _get_index_paths()
    if _faiss_index is not None:
        faiss.write_index(_faiss_index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(_metadata_store, f)
        logger.info(f"Saved FAISS index ({_faiss_index.ntotal} vectors)")


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate L2-normalized embeddings for a list of texts.
    Returns shape (N, 384) float32 array.
    """
    model = _get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # L2 normalize for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    return (embeddings / norms).astype(np.float32)


def add_chunks_to_index(chunks: list, document_id: str) -> int:
    """
    Embed chunks and add them to the FAISS index.
    Returns the number of vectors added.
    """
    global _faiss_index, _metadata_store

    import faiss

    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]

    # Initialize index on first use
    if _faiss_index is None:
        _load_index()
    if _faiss_index is None:
        _faiss_index = faiss.IndexFlatIP(dim)
        logger.info(f"Created new FAISS IndexFlatIP (dim={dim})")

    _faiss_index.add(embeddings)

    # Store metadata in parallel
    for chunk in chunks:
        _metadata_store.append({
            "document_id": document_id,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        })

    _save_index()
    return len(chunks)


def search_index(query: str, top_k: int = 5, document_id: Optional[str] = None) -> list[dict]:
    """
    Retrieve top-k most similar chunks for a query.

    Args:
        query: User question.
        top_k: Number of results to return.
        document_id: If provided, filter results to this document only.

    Returns:
        List of dicts with text, source, chunk_index, similarity_score.
    """
    global _faiss_index, _metadata_store

    if _faiss_index is None:
        _load_index()
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return []

    query_embedding = embed_texts([query])  # shape (1, 384)

    # Search more than top_k if filtering by document_id
    search_k = top_k * 5 if document_id else top_k
    search_k = min(search_k, _faiss_index.ntotal)

    scores, indices = _faiss_index.search(query_embedding, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = _metadata_store[idx]
        if document_id and meta["document_id"] != document_id:
            continue
        results.append({
            "text": meta["text"],
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "similarity_score": float(score),
            "document_id": meta["document_id"],
        })
        if len(results) >= top_k:
            break

    return results
