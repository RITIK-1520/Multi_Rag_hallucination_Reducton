"""
evaluate.py – standalone script to run the full evaluation suite.

Usage:
    python evaluate.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_documents
from src.chunking import chunk_all_documents
from src.embeddings import generate_embeddings
from src.vector_store import VectorStore
from src.orchestrator import Orchestrator
from src.evaluation import (
    load_evaluation_questions,
    run_evaluation,
    save_results_csv,
    save_results_json,
)
from src.config import is_api_available, OUTPUTS_DIR
from src.utils import ensure_dir


def build_or_load_index() -> VectorStore:
    """Build (or load cached) FAISS index."""
    vs = VectorStore()
    if vs.load():
        print("[evaluate] Loaded existing FAISS index")
        return vs
    print("[evaluate] Building FAISS index from documents...")
    docs = load_documents()
    chunks = chunk_all_documents(docs)
    texts = [c["text"] for c in chunks]
    embs = generate_embeddings(texts)
    vs.build_index(chunks, embs)
    vs.save()
    return vs


def main():
    print("=" * 60)
    print("  Multi-Agent RAG – Evaluation Runner")
    print("=" * 60)
    mode = "Live Gemini Mode" if is_api_available() else "Demo / Offline Mode"
    print(f"  Mode: {mode}\n")

    # Build / load index
    vs = build_or_load_index()

    # Load questions
    questions = load_evaluation_questions()
    print(f"  Loaded {len(questions)} evaluation questions\n")

    # Run evaluation
    orchestrator = Orchestrator(vs)
    eval_output = run_evaluation(orchestrator, questions)

    results = eval_output["results"]
    metrics = eval_output["metrics"]

    # Save outputs
    ensure_dir(OUTPUTS_DIR)
    csv_path = save_results_csv(results)
    json_path = save_results_json(eval_output)

    # Generate demo report
    generate_demo_report(results, metrics)

    # Print summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total questions:            {metrics['total_questions']}")
    print(f"  Multi-Agent avg support:    {metrics['multi_avg_support_pct']:.1f}%")
    print(f"  Multi-Agent avg risk:       {metrics['multi_avg_hallucination_risk']:.4f}")
    print(f"  Multi-Agent verified:       {metrics['multi_verified_count']}")
    print(f"  Multi-Agent abstained:      {metrics['multi_abstained_count']}")
    print(f"  Multi-Agent regenerated:    {metrics['multi_regenerated_count']}")
    print(f"  Multi-Agent avg time:       {metrics['multi_avg_time']:.3f}s")
    print(f"  Basic RAG avg confidence:   {metrics['basic_avg_confidence']:.4f}")
    print(f"  Basic RAG avg time:         {metrics['basic_avg_time']:.3f}s")
    print(f"\n  CSV saved to:  {csv_path}")
    print(f"  JSON saved to: {json_path}")
    print("=" * 60)


def generate_demo_report(results, metrics):
    """Generate the demo_report.md file from real evaluation results."""
    ensure_dir(OUTPUTS_DIR)
    report_path = os.path.join(OUTPUTS_DIR, "demo_report.md")

    lines = [
        "# Multi-Agent RAG – Demo Report",
        "",
        f"**Generated automatically from evaluation results.**",
        "",
        "---",
        "",
    ]

    # Pick representative examples
    demos = []
    for r in results:
        if r["multi_verification_status"] == "VERIFIED" and not demos:
            demos.append(("Successful Verified Answer", r))
        elif r["multi_verification_status"] == "INSUFFICIENT_EVIDENCE" and len(demos) < 4:
            demos.append(("Insufficient Evidence / Abstention", r))
        elif r["type"] == "hallucination_trigger" and len(demos) < 4:
            demos.append(("Hallucination Trigger – Cautious Response", r))
        elif r["type"] == "tricky" and len(demos) < 4:
            demos.append(("Tricky Question", r))

    # Fill remaining slots
    for r in results:
        if len(demos) >= 4:
            break
        if r not in [d[1] for d in demos]:
            demos.append(("Additional Example", r))

    for title, r in demos[:4]:
        lines.extend([
            f"## {title}",
            "",
            f"**Question:** {r['question']}",
            "",
            f"**Basic RAG Answer:** {r['basic_answer'][:300]}",
            "",
            f"**Multi-Agent RAG Answer:** {r['multi_answer'][:300]}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Evidence Confidence | {r['multi_evidence_confidence']:.4f} |",
            f"| Support Percentage | {r['multi_support_pct']}% |",
            f"| Hallucination Risk | {r['multi_hallucination_risk']} |",
            f"| Verification Status | {r['multi_verification_status']} |",
            f"| Processing Time | {r['multi_time']:.3f}s |",
            "",
            "---",
            "",
        ])

    # Summary
    lines.extend([
        "## Evaluation Summary",
        "",
        f"| Metric | Basic RAG | Multi-Agent RAG |",
        f"|--------|-----------|-----------------|",
        f"| Avg Confidence | {metrics['basic_avg_confidence']:.4f} | {metrics['multi_avg_evidence_confidence']:.4f} |",
        f"| Avg Support % | N/A | {metrics['multi_avg_support_pct']:.1f}% |",
        f"| Avg Hallucination Risk | Unknown | {metrics['multi_avg_hallucination_risk']:.4f} |",
        f"| Verified Answers | N/A | {metrics['multi_verified_count']} |",
        f"| Abstained | N/A | {metrics['multi_abstained_count']} |",
        f"| Avg Time | {metrics['basic_avg_time']:.3f}s | {metrics['multi_avg_time']:.3f}s |",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[evaluate] Demo report saved to {report_path}")


if __name__ == "__main__":
    main()
