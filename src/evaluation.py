"""
Evaluation module – runs predefined questions through both pipelines
and computes comparison metrics.
"""

import json
import os
import time
import pandas as pd

from src.config import EVALUATION_DIR, OUTPUTS_DIR
from src.orchestrator import Orchestrator
from src.utils import ensure_dir


def load_evaluation_questions(path: str | None = None) -> list[dict]:
    """Load evaluation questions from JSON file."""
    if path is None:
        path = os.path.join(EVALUATION_DIR, "evaluation_questions.json")
    if not os.path.exists(path):
        print(f"[Evaluation] File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", data) if isinstance(data, dict) else data


def run_evaluation(orchestrator: Orchestrator,
                   questions: list[dict] | None = None,
                   top_k: int = 5,
                   temperature: float = 0.3,
                   hallucination_threshold: float = 0.4,
                   evidence_threshold: float = 0.35,
                   max_attempts: int = 2) -> dict:
    """
    Run every question through both Basic RAG and Multi-Agent RAG.
    Returns a dict with individual results and aggregate metrics.
    """
    if questions is None:
        questions = load_evaluation_questions()

    results = []

    for i, q in enumerate(questions):
        query = q["question"]
        q_type = q.get("type", "general")
        expected = q.get("expected_behavior", "")
        has_ground_truth = "ground_truth" in q

        print(f"[Eval] ({i+1}/{len(questions)}) {query[:60]}...")

        # ── Basic RAG ──
        basic = orchestrator.run_basic_rag(query, top_k, temperature)

        # ── Multi-Agent RAG ──
        multi = orchestrator.run_multi_agent_rag(
            query, top_k, temperature,
            hallucination_threshold, evidence_threshold, max_attempts
        )

        row = {
            "question_id": i + 1,
            "question": query,
            "type": q_type,
            "expected_behavior": expected,
            # Basic RAG
            "basic_answer": basic["answer"],
            "basic_sources": ", ".join(basic.get("sources", [])),
            "basic_confidence": basic.get("evidence_confidence", 0),
            "basic_time": basic["processing_time"],
            # Multi-Agent RAG
            "multi_answer": multi["answer"],
            "multi_sources": ", ".join(multi.get("sources", [])),
            "multi_evidence_confidence": multi.get("evidence_confidence", 0),
            "multi_support_pct": multi.get("support_percentage", 0),
            "multi_hallucination_risk": multi.get("hallucination_risk", 0),
            "multi_verification_status": multi.get("verification_status", ""),
            "multi_supported_claims": sum(
                1 for c in multi.get("claims", []) if c.get("status") == "SUPPORTED"),
            "multi_unsupported_claims": sum(
                1 for c in multi.get("claims", []) if c.get("status") == "UNSUPPORTED"),
            "multi_attempts": multi.get("attempt", 0),
            "multi_time": multi["processing_time"],
        }
        results.append(row)

    # ── Aggregate metrics ──
    metrics = _compute_aggregate_metrics(results)
    return {"results": results, "metrics": metrics}


def _compute_aggregate_metrics(results: list[dict]) -> dict:
    """Compute aggregate comparison metrics."""
    n = len(results)
    if n == 0:
        return {}

    # Multi-Agent metrics
    support_pcts = [r["multi_support_pct"] for r in results if r["multi_support_pct"] is not None]
    risk_vals = [r["multi_hallucination_risk"] for r in results if r["multi_hallucination_risk"] is not None]
    ev_confs = [r["multi_evidence_confidence"] for r in results]

    verified = sum(1 for r in results if r["multi_verification_status"] == "VERIFIED")
    abstained = sum(1 for r in results
                    if r["multi_verification_status"] == "INSUFFICIENT_EVIDENCE")
    regenerated = sum(1 for r in results if r["multi_attempts"] > 1)
    rejected = sum(1 for r in results if r["multi_verification_status"] in ("REJECTED", "HIGH_RISK"))

    avg = lambda lst: sum(lst) / len(lst) if lst else 0

    return {
        "total_questions": n,
        # Multi-Agent
        "multi_avg_support_pct": round(avg(support_pcts), 2),
        "multi_avg_hallucination_risk": round(avg(risk_vals), 4),
        "multi_avg_evidence_confidence": round(avg(ev_confs), 4),
        "multi_avg_time": round(avg([r["multi_time"] for r in results]), 3),
        "multi_verified_count": verified,
        "multi_abstained_count": abstained,
        "multi_regenerated_count": regenerated,
        "multi_rejected_count": rejected,
        # Basic RAG
        "basic_avg_confidence": round(avg([r["basic_confidence"] for r in results]), 4),
        "basic_avg_time": round(avg([r["basic_time"] for r in results]), 3),
        # Basic RAG has no verification, so hallucination rate is unknown
        "basic_verification": "None (no verification in basic RAG)",
    }


def save_results_csv(results: list[dict], path: str | None = None):
    """Save evaluation results to CSV."""
    if path is None:
        ensure_dir(OUTPUTS_DIR)
        path = os.path.join(OUTPUTS_DIR, "evaluation_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"[Evaluation] Saved CSV to {path}")
    return path


def save_results_json(eval_output: dict, path: str | None = None):
    """Save full evaluation output (results + metrics) to JSON."""
    if path is None:
        ensure_dir(OUTPUTS_DIR)
        path = os.path.join(OUTPUTS_DIR, "sample_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, default=str)
    print(f"[Evaluation] Saved JSON to {path}")
    return path
