"""Tests for retrieval pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_documents
from src.chunking import chunk_all_documents
from src.embeddings import generate_embeddings, generate_query_embedding
from src.vector_store import VectorStore


def test_document_loading():
    docs = load_documents()
    assert len(docs) >= 8, f"Expected >=8 docs, got {len(docs)}"
    for d in docs:
        assert "document_name" in d
        assert "text" in d
        assert len(d["text"]) > 100
    print("[PASS] test_document_loading")


def test_chunking():
    docs = load_documents()
    chunks = chunk_all_documents(docs)
    assert len(chunks) > 20, f"Expected >20 chunks, got {len(chunks)}"
    for c in chunks:
        assert "chunk_id" in c
        assert "document_name" in c
        assert "text" in c
        assert len(c["text"]) > 10
    print("[PASS] test_chunking")


def test_embeddings():
    texts = ["Hello world", "Test embedding generation"]
    embs = generate_embeddings(texts)
    assert embs.shape[0] == 2
    assert embs.shape[1] == 384  # MiniLM dimension
    print("[PASS] test_embeddings")


def test_vector_store_build_and_search():
    docs = load_documents()
    chunks = chunk_all_documents(docs)
    texts = [c["text"] for c in chunks]
    embs = generate_embeddings(texts)

    vs = VectorStore()
    vs.build_index(chunks, embs)
    assert vs.num_vectors == len(chunks)

    q_emb = generate_query_embedding("What is artificial intelligence?")
    results = vs.search(q_emb, top_k=3)
    assert len(results) > 0
    assert "similarity_score" in results[0]
    assert results[0]["similarity_score"] > 0
    print("[PASS] test_vector_store_build_and_search")


def test_vector_store_save_load():
    import tempfile, shutil
    docs = load_documents()
    chunks = chunk_all_documents(docs)
    embs = generate_embeddings([c["text"] for c in chunks[:5]])

    temp_dir = tempfile.mkdtemp()
    try:
        vs = VectorStore()
        vs.build_index(chunks[:5], embs)
        vs.save(temp_dir)

        vs2 = VectorStore()
        ok = vs2.load(temp_dir)
        assert ok, "Failed to load saved index"
        assert vs2.num_vectors == 5
        print("[PASS] test_vector_store_save_load")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_document_loading()
    test_chunking()
    test_embeddings()
    test_vector_store_build_and_search()
    test_vector_store_save_load()
    print("\n=== All retrieval tests passed ===")
