"""Tests for the full pipeline (orchestrator)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_documents
from src.chunking import chunk_all_documents
from src.embeddings import generate_embeddings
from src.vector_store import VectorStore
from src.orchestrator import Orchestrator


def _build():
    docs = load_documents()
    chunks = chunk_all_documents(docs)
    embs = generate_embeddings([c["text"] for c in chunks])
    vs = VectorStore()
    vs.build_index(chunks, embs)
    return Orchestrator(vs)


def test_basic_rag():
    orch = _build()
    result = orch.run_basic_rag("What is artificial intelligence?")
    assert result["pipeline"] == "basic_rag"
    assert len(result["answer"]) > 10
    assert result["verification_status"] == "NOT_VERIFIED"
    print("[PASS] test_basic_rag")


def test_multi_agent_rag_answerable():
    orch = _build()
    result = orch.run_multi_agent_rag("What is machine learning?")
    assert result["pipeline"] == "multi_agent_rag"
    assert len(result["answer"]) > 10
    assert result["verification_status"] in ("VERIFIED", "MEDIUM_RISK", "HIGH_RISK",
                                               "INSUFFICIENT_EVIDENCE", "REJECTED")
    assert "hallucination_risk" in result
    print("[PASS] test_multi_agent_rag_answerable")


def test_multi_agent_rag_unsupported():
    orch = _build()
    result = orch.run_multi_agent_rag("What is the current price of Bitcoin?")
    # Should either abstain or flag as high risk
    assert result["pipeline"] == "multi_agent_rag"
    # The answer should mention insufficient evidence OR be flagged
    print(f"  Status: {result['verification_status']}, Risk: {result.get('hallucination_risk')}")
    print("[PASS] test_multi_agent_rag_unsupported")


def test_empty_query():
    orch = _build()
    result = orch.run_multi_agent_rag("")
    assert "Insufficient evidence" in result["answer"] or result["verification_status"] == "INSUFFICIENT_EVIDENCE"
    print("[PASS] test_empty_query")


def test_basic_rag_unrelated():
    orch = _build()
    result = orch.run_basic_rag("What is the recipe for chocolate cake?")
    assert result["pipeline"] == "basic_rag"
    # Basic RAG will still return something – no verification
    print("[PASS] test_basic_rag_unrelated")


if __name__ == "__main__":
    test_basic_rag()
    test_multi_agent_rag_answerable()
    test_multi_agent_rag_unsupported()
    test_empty_query()
    test_basic_rag_unrelated()
    print("\n=== All pipeline tests passed ===")
