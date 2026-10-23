"""Tests for hallucination checker module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hallucination_checker import compute_hallucination_score


def test_all_supported():
    critic = {
        "supported_count": 3,
        "partially_supported_count": 0,
        "unsupported_count": 0,
        "contradiction_count": 0,
    }
    chunks = [{"similarity_score": 0.8}, {"similarity_score": 0.7}]
    result = compute_hallucination_score(critic, 0.8, chunks)
    assert result["hallucination_risk"] < 0.3
    assert result["verification_status"] == "VERIFIED"
    print("[PASS] test_all_supported")


def test_mixed_claims():
    critic = {
        "supported_count": 2,
        "partially_supported_count": 1,
        "unsupported_count": 2,
        "contradiction_count": 0,
    }
    chunks = [{"similarity_score": 0.5}]
    result = compute_hallucination_score(critic, 0.5, chunks)
    assert result["hallucination_risk"] > 0.3
    assert result["total_claims"] == 5
    print("[PASS] test_mixed_claims")


def test_contradiction():
    critic = {
        "supported_count": 1,
        "partially_supported_count": 0,
        "unsupported_count": 0,
        "contradiction_count": 1,
    }
    chunks = [{"similarity_score": 0.6}]
    result = compute_hallucination_score(critic, 0.6, chunks)
    assert result["verification_status"] == "CONTRADICTED"
    print("[PASS] test_contradiction")


def test_no_claims():
    critic = {
        "supported_count": 0,
        "partially_supported_count": 0,
        "unsupported_count": 0,
        "contradiction_count": 0,
    }
    result = compute_hallucination_score(critic, 0.0, [])
    assert result["verification_status"] == "NO_CLAIMS"
    assert result["hallucination_risk"] == 0.0
    print("[PASS] test_no_claims")


def test_high_risk():
    critic = {
        "supported_count": 0,
        "partially_supported_count": 0,
        "unsupported_count": 5,
        "contradiction_count": 0,
    }
    chunks = [{"similarity_score": 0.2}]
    result = compute_hallucination_score(critic, 0.2, chunks)
    assert result["hallucination_risk"] > 0.6
    assert result["verification_status"] == "HIGH_RISK"
    print("[PASS] test_high_risk")


if __name__ == "__main__":
    test_all_supported()
    test_mixed_claims()
    test_contradiction()
    test_no_claims()
    test_high_risk()
    print("\n=== All hallucination checker tests passed ===")
