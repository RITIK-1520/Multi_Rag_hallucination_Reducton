"""Tests for individual agents."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import (
    QueryAnalyzerAgent,
    RetrieverAgent,
    EvidenceVerifierAgent,
    AnswerGeneratorAgent,
    HallucinationCriticAgent,
)
from src.data_loader import load_documents
from src.chunking import chunk_all_documents
from src.embeddings import generate_embeddings
from src.vector_store import VectorStore


def _build_vs():
    docs = load_documents()
    chunks = chunk_all_documents(docs)
    embs = generate_embeddings([c["text"] for c in chunks])
    vs = VectorStore()
    vs.build_index(chunks, embs)
    return vs


def test_query_analyzer():
    agent = QueryAnalyzerAgent()
    result = agent.run("What is machine learning?")
    assert result["status"] == "completed"
    assert result["original_query"] == "What is machine learning?"
    assert "intent" in result
    assert "keywords" in result
    print("[PASS] test_query_analyzer")


def test_query_analyzer_empty():
    agent = QueryAnalyzerAgent()
    result = agent.run("")
    assert result["status"] == "error"
    print("[PASS] test_query_analyzer_empty")


def test_retriever():
    vs = _build_vs()
    agent = RetrieverAgent(vs)
    result = agent.run("What is deep learning?", top_k=3)
    assert result["status"] == "completed"
    assert len(result["results"]) > 0
    assert result["results"][0]["similarity_score"] > 0
    print("[PASS] test_retriever")


def test_evidence_verifier():
    agent = EvidenceVerifierAgent()
    chunks = [
        {"text": "Deep learning is a subset of machine learning", "similarity_score": 0.7,
         "document_name": "deep_learning", "chunk_id": "dl_0", "source": "test"},
        {"text": "Irrelevant chunk", "similarity_score": 0.1,
         "document_name": "other", "chunk_id": "o_0", "source": "test"},
    ]
    result = agent.run("What is deep learning?", chunks)
    assert result["status"] == "completed"
    assert len(result["supporting_chunks"]) >= 1
    assert result["evidence_status"] in ("sufficient", "weak", "insufficient")
    print("[PASS] test_evidence_verifier")


def test_evidence_verifier_empty():
    agent = EvidenceVerifierAgent()
    result = agent.run("anything", [])
    assert result["evidence_status"] == "insufficient"
    print("[PASS] test_evidence_verifier_empty")


def test_answer_generator():
    agent = AnswerGeneratorAgent()
    chunks = [
        {"text": "AI is a branch of computer science.", "similarity_score": 0.8,
         "document_name": "artificial_intelligence", "chunk_id": "ai_0", "source": "t"},
    ]
    result = agent.run("What is AI?", chunks, "sufficient")
    assert result["status"] == "completed"
    assert len(result["answer"]) > 10
    print("[PASS] test_answer_generator")


def test_answer_generator_insufficient():
    agent = AnswerGeneratorAgent()
    result = agent.run("Anything", [], "insufficient")
    assert "Insufficient evidence" in result["answer"]
    assert result["status"] == "abstained"
    print("[PASS] test_answer_generator_insufficient")


def test_hallucination_critic():
    agent = HallucinationCriticAgent()
    answer = "Deep learning is a subset of machine learning. It uses neural networks."
    chunks = [
        {"text": "Deep learning is a subset of Machine Learning that uses artificial neural networks",
         "document_name": "dl", "chunk_id": "dl_0", "source": "t"},
    ]
    result = agent.run(answer, chunks)
    assert result["status"] == "completed"
    assert "claims" in result
    assert "hallucination_risk" in result
    print("[PASS] test_hallucination_critic")


if __name__ == "__main__":
    test_query_analyzer()
    test_query_analyzer_empty()
    test_retriever()
    test_evidence_verifier()
    test_evidence_verifier_empty()
    test_answer_generator()
    test_answer_generator_insufficient()
    test_hallucination_critic()
    print("\n=== All agent tests passed ===")
