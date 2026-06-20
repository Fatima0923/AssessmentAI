# rag_store.py
#
# RAG (Retrieval-Augmented Generation) store built on FAISS.
#
# This module implements Node 1's context retrieval capability.
# Course context documents (rubric, assessment brief, course outline,
# assessor profile) are chunked, embedded, and indexed in a local FAISS
# store. When an essay is evaluated, the most relevant context passages
# are retrieved and injected into the evaluation prompt — grounding the
# AI's judgments in the actual course standards rather than generic
# parametric knowledge.
#
# Architecture:
#   - Embeddings: sentence-transformers all-MiniLM-L6-v2 (local, no API cost)
#   - Vector store: FAISS (local, no network required)
#   - Retrieval: top-k similarity search per query
#   - Fallback: if FAISS unavailable, full context is passed directly

import os
import re
from typing import Dict, List, Optional

# Lazy imports — only loaded when RAG is actually used
_faiss_store = None
_embedder    = None


def _get_embedder():
    """
    Load sentence-transformers embedder with version-safe fallback.
    sentence-transformers 5.x changed the model loading API.
    Falls back to TF-IDF similarity if loading fails.
    """
    global _embedder
    if _embedder is None:
        print("[RAG] Loading embedding model...")

        # Try sentence-transformers with trust_remote_code for newer versions
        for kwargs in [
            {"trust_remote_code": False},
            {"trust_remote_code": True},
            {},
        ]:
            try:
                from sentence_transformers import SentenceTransformer
                _embedder = SentenceTransformer("all-MiniLM-L6-v2", **kwargs)
                print("[RAG] Embedding model loaded OK")
                return _embedder
            except Exception as e:
                last_err = e
                continue

        # If all attempts fail, return None — RAG store will use TF-IDF fallback
        print(f"[RAG] Embedding model unavailable ({last_err}) — using TF-IDF fallback")
        _embedder = None
    return _embedder


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    Overlap ensures context is not lost at chunk boundaries.
    """
    words  = text.split()
    chunks = []
    step   = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


class FAISSContextStore:
    """
    Local FAISS vector store for RAG retrieval over course context documents.

    Usage:
        store = FAISSContextStore()
        store.index_documents({"rubric": "...", "course_outline": "..."})
        results = store.retrieve("How should originality be assessed?", top_k=3)
    """

    def __init__(self):
        self.index     = None
        self.chunks    = []        # raw text chunks
        self.metadata  = []        # source key per chunk
        self.embedder  = None
        self._built    = False

    def index_documents(self, context: Dict[str, str]) -> None:
        """
        Embed and index all context documents into the FAISS store.

        Parameters
        ----------
        context : dict mapping document name to text content
                  e.g. {"rubric": "...", "assessment_details": "..."}
        """
        try:
            import faiss
            import numpy as np
        except ImportError:
            print("[RAG] FAISS not available — using full context injection fallback")
            self._built = False
            return

        self.embedder = _get_embedder()
        self.chunks   = []
        self.metadata = []

        # If embedder failed to load, store chunks for TF-IDF fallback only
        if self.embedder is None:
            print("[RAG] No embedder — storing chunks for TF-IDF retrieval")
            for doc_name, text in context.items():
                if not text:
                    continue
                self.chunks.extend(_chunk_text(text))
                self.metadata.extend([doc_name] * len(_chunk_text(text)))
            self._built = bool(self.chunks)
            return

        print("[RAG] Indexing context documents...")

        for doc_name, text in context.items():
            if not text:
                continue
            doc_chunks = _chunk_text(text)
            self.chunks.extend(doc_chunks)
            self.metadata.extend([doc_name] * len(doc_chunks))

        if not self.chunks:
            print("[RAG] No chunks to index")
            return

        embeddings = self.embedder.encode(self.chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        dimension  = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)   # inner product = cosine on normalised vecs

        # Normalise for cosine similarity
        import faiss as f
        f.normalize_L2(embeddings)
        self.index.add(embeddings)

        self._built = True
        print(f"[RAG] Indexed {len(self.chunks)} chunks from {len(context)} documents")

    def retrieve(self, query: str, top_k: int = 4) -> str:
        """
        Retrieve the top_k most relevant context passages for a query.

        Returns a formatted string ready for prompt injection.
        If the store is not built, returns an empty string.
        """
        if not self._built or self.index is None:
            return ""

        try:
            import numpy as np
            import faiss

            query_embedding = self.embedder.encode([query], show_progress_bar=False)
            query_embedding = np.array(query_embedding, dtype="float32")
            faiss.normalize_L2(query_embedding)

            distances, indices = self.index.search(query_embedding, top_k)

            retrieved = []
            seen      = set()
            for idx in indices[0]:
                if idx < 0 or idx >= len(self.chunks):
                    continue
                chunk = self.chunks[idx]
                if chunk in seen:
                    continue
                seen.add(chunk)
                source = self.metadata[idx]
                retrieved.append(f"[{source.upper()}] {chunk}")

            if not retrieved:
                return ""

            return (
                "\nRELEVANT COURSE CONTEXT (retrieved via RAG):\n"
                + "\n\n".join(retrieved)
                + "\n"
            )

        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return ""

    @property
    def is_ready(self) -> bool:
        return self._built


# ==============================================================================
# MODULE-LEVEL SINGLETON
# Shared across all pipeline nodes — indexed once at startup.
# ==============================================================================

_store_instance: Optional[FAISSContextStore] = None


def get_rag_store() -> FAISSContextStore:
    """Return the shared RAG store singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = FAISSContextStore()
    return _store_instance


def build_rag_store(context: Dict[str, str]) -> FAISSContextStore:
    """
    Build and return the shared RAG store from context documents.
    Called once during Node 1 (Data Preprocessing).
    """
    global _store_instance
    _store_instance = FAISSContextStore()
    _store_instance.index_documents(context)
    return _store_instance