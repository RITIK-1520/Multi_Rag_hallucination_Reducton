"""
Embedding module – generates vector embeddings using Sentence Transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

# Module-level cache so the model is loaded only once per process.
_model_cache: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return (and cache) the sentence-transformer model."""
    global _model_cache
    if _model_cache is None:
        print(f"[Embeddings] Loading model: {EMBEDDING_MODEL_NAME}")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[Embeddings] Model loaded successfully")
    return _model_cache


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False,
                              convert_to_numpy=True)
    return embeddings.astype("float32")


def generate_query_embedding(query: str) -> np.ndarray:
    """Generate embedding for a single query string."""
    return generate_embeddings([query])[0]
