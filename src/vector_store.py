"""
Vector store module – FAISS index creation, saving, loading, and search.
"""

import os
import json
import numpy as np
import faiss

from src.config import FAISS_INDEX_DIR, EMBEDDING_DIMENSION
from src.utils import ensure_dir

INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "index.faiss")
META_FILE = os.path.join(FAISS_INDEX_DIR, "metadata.json")


class VectorStore:
    """Wrapper around a FAISS flat (L2) index with chunk metadata."""

    def __init__(self):
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[dict] = []

    # ── Build ──
    def build_index(self, chunks: list[dict], embeddings: np.ndarray):
        """Create a new FAISS index from chunk embeddings."""
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.chunks = chunks
        print(f"[VectorStore] Built index with {self.index.ntotal} vectors (dim={dim})")

    # ── Persist ──
    def save(self, directory: str = FAISS_INDEX_DIR):
        """Save the FAISS index and metadata to disk."""
        ensure_dir(directory)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)
        print(f"[VectorStore] Saved index and metadata to {directory}")

    def load(self, directory: str = FAISS_INDEX_DIR) -> bool:
        """Load a previously saved index. Returns True on success."""
        idx_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "metadata.json")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            print("[VectorStore] No saved index found")
            return False
        try:
            self.index = faiss.read_index(idx_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            print(f"[VectorStore] Loaded index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"[VectorStore] Error loading index: {e}")
            return False

    # ── Search ──
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search the index for the top-k nearest chunks.

        Returns list of dicts, each containing the chunk metadata plus
        'similarity_score' (cosine-like, converted from L2 distance).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = query_embedding.reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            # Convert L2 distance to a similarity score in [0, 1]
            chunk["similarity_score"] = round(float(1 / (1 + dist)), 4)
            results.append(chunk)

        return results

    # ── Info ──
    @property
    def num_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def index_exists(self, directory: str = FAISS_INDEX_DIR) -> bool:
        """Check whether a saved index is available on disk."""
        return (os.path.exists(os.path.join(directory, "index.faiss"))
                and os.path.exists(os.path.join(directory, "metadata.json")))
