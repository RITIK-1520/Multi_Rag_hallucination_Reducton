"""
Hallucination checker – standalone utility that computes the overall
hallucination risk score from critic output.

Formula:
    Support Score = (supported + 0.5 * partially_supported) / total_claims
    Hallucination Risk = 1 - Support Score

Factors combined:
    - evidence_confidence   (from Evidence Verifier)
    - support_percentage    (from Hallucination Critic)
    - retrieval_relevance   (average similarity of supporting chunks)
"""


def compute_hallucination_score(critic_result: dict,
                                 evidence_confidence: float,
                                 supporting_chunks: list[dict]) -> dict:
    """
    Compute a deterministic, explainable hallucination risk score.

    Returns a dict with all sub-scores and the final risk.
    """
    total_claims = (
        critic_result.get("supported_count", 0)
        + critic_result.get("partially_supported_count", 0)
        + critic_result.get("unsupported_count", 0)
        + critic_result.get("contradiction_count", 0)
    )

    if total_claims == 0:
        return {
            "total_claims": 0,
            "support_score": 0.0,
            "evidence_confidence": evidence_confidence,
            "retrieval_relevance": 0.0,
            "hallucination_risk": 0.0,
            "verification_status": "NO_CLAIMS",
            "explanation": "No claims to verify (answer was likely abstained).",
        }

    # ── Factor 1: Support Score ──
    support_score = (
        critic_result.get("supported_count", 0)
        + 0.5 * critic_result.get("partially_supported_count", 0)
    ) / total_claims

    # ── Factor 2: Evidence Confidence (passed in) ──
    ev_conf = evidence_confidence

    # ── Factor 3: Retrieval Relevance ──
    if supporting_chunks:
        retrieval_relevance = sum(
            c.get("similarity_score", 0) for c in supporting_chunks
        ) / len(supporting_chunks)
    else:
        retrieval_relevance = 0.0

    # ── Combined Hallucination Risk ──
    # Weighted combination: 50% claim support, 25% evidence confidence, 25% retrieval relevance
    combined_confidence = (
        0.50 * support_score
        + 0.25 * ev_conf
        + 0.25 * retrieval_relevance
    )
    hallucination_risk = round(1.0 - combined_confidence, 4)
    hallucination_risk = max(0.0, min(1.0, hallucination_risk))

    # ── Verification Status ──
    if critic_result.get("contradiction_count", 0) > 0:
        status = "CONTRADICTED"
    elif hallucination_risk > 0.6:
        status = "HIGH_RISK"
    elif hallucination_risk > 0.35:
        status = "MEDIUM_RISK"
    else:
        status = "VERIFIED"

    return {
        "total_claims": total_claims,
        "supported_count": critic_result.get("supported_count", 0),
        "partially_supported_count": critic_result.get("partially_supported_count", 0),
        "unsupported_count": critic_result.get("unsupported_count", 0),
        "contradiction_count": critic_result.get("contradiction_count", 0),
        "support_score": round(support_score, 4),
        "evidence_confidence": round(ev_conf, 4),
        "retrieval_relevance": round(retrieval_relevance, 4),
        "hallucination_risk": hallucination_risk,
        "support_percentage": round(support_score * 100, 1),
        "verification_status": status,
        "explanation": (
            f"Support Score={support_score:.2f} (weight 50%), "
            f"Evidence Confidence={ev_conf:.2f} (weight 25%), "
            f"Retrieval Relevance={retrieval_relevance:.2f} (weight 25%) → "
            f"Hallucination Risk={hallucination_risk:.2f}, Status={status}"
        ),
    }
