"""
Orchestrator – Agent 6.
Coordinates all agents, implements the regeneration loop, and
provides both Basic RAG and Multi-Agent RAG pipelines.
"""

import time

from src.agents import (
    QueryAnalyzerAgent,
    RetrieverAgent,
    EvidenceVerifierAgent,
    AnswerGeneratorAgent,
    HallucinationCriticAgent,
)
from src.hallucination_checker import compute_hallucination_score
from src.vector_store import VectorStore


class Orchestrator:
    """Main pipeline coordinator for Multi-Agent RAG."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.query_analyzer = QueryAnalyzerAgent()
        self.retriever = RetrieverAgent(vector_store)
        self.evidence_verifier = EvidenceVerifierAgent()
        self.answer_generator = AnswerGeneratorAgent()
        self.hallucination_critic = HallucinationCriticAgent()

    # ═══════════════════════════════════════════════════
    # Basic RAG pipeline (baseline)
    # ═══════════════════════════════════════════════════
    def run_basic_rag(self, query: str, top_k: int = 5,
                      temperature: float = 0.3) -> dict:
        """
        Simple RAG: retrieve → generate.  No verification, no critic.
        """
        pipeline_start = time.time()

        # Retrieve
        retrieval = self.retriever.run(query, top_k=top_k)
        chunks = retrieval["results"]

        # Generate answer directly from all retrieved chunks
        answer_result = self.answer_generator.run(
            query, chunks, "sufficient", temperature
        )

        # Compute a simple retrieval-only confidence
        if chunks:
            avg_sim = sum(c["similarity_score"] for c in chunks) / len(chunks)
        else:
            avg_sim = 0.0

        return {
            "query": query,
            "answer": answer_result["answer"],
            "sources": answer_result["sources_used"],
            "retrieved_chunks": chunks,
            "retrieval_confidence": round(avg_sim, 4),
            "pipeline": "basic_rag",
            "processing_time": round(time.time() - pipeline_start, 3),
            # No verification data
            "evidence_confidence": round(avg_sim, 4),
            "hallucination_risk": None,
            "support_percentage": None,
            "verification_status": "NOT_VERIFIED",
            "claims": [],
            "agent_trace": [
                {"agent": "Retriever", "status": "completed",
                 "time": retrieval["execution_time"]},
                {"agent": "Answer Generator", "status": "completed",
                 "time": answer_result["execution_time"]},
            ],
        }

    # ═══════════════════════════════════════════════════
    # Multi-Agent RAG pipeline (proposed system)
    # ═══════════════════════════════════════════════════
    def run_multi_agent_rag(self, query: str, top_k: int = 5,
                            temperature: float = 0.3,
                            hallucination_threshold: float = 0.4,
                            evidence_threshold: float = 0.35,
                            max_attempts: int = 2) -> dict:
        """
        Full multi-agent pipeline with verification and regeneration loop.
        """
        pipeline_start = time.time()
        agent_trace: list[dict] = []

        # ── Step 1: Query Analysis ──
        analysis = self.query_analyzer.run(query, temperature)
        agent_trace.append({
            "agent": "Query Analyzer",
            "status": analysis.get("status", "completed"),
            "time": analysis.get("execution_time", 0),
            "output_summary": f"Intent: {analysis.get('intent')}, "
                              f"Keywords: {analysis.get('keywords', [])}",
        })

        if analysis.get("status") == "error":
            return self._abstain_result(query, agent_trace, pipeline_start,
                                        "Query analysis failed: " + analysis.get("error", ""))

        # ── Step 2: Retrieval ──
        retrieval = self.retriever.run(
            query, sub_queries=analysis.get("sub_queries"), top_k=top_k
        )
        agent_trace.append({
            "agent": "Retriever",
            "status": retrieval["status"],
            "time": retrieval["execution_time"],
            "output_summary": f"Retrieved {retrieval['num_results']} chunks",
        })

        # ── Step 3: Evidence Verification ──
        verification = self.evidence_verifier.run(
            query, retrieval["results"], evidence_threshold, temperature
        )
        agent_trace.append({
            "agent": "Evidence Verifier",
            "status": verification["status"],
            "time": verification["execution_time"],
            "output_summary": (
                f"Status: {verification['evidence_status']}, "
                f"Confidence: {verification['confidence_score']:.2f}, "
                f"Supporting: {len(verification['supporting_chunks'])}, "
                f"Rejected: {len(verification['rejected_chunks'])}"
            ),
        })

        if verification["evidence_status"] == "insufficient":
            agent_trace.append({
                "agent": "Final Decision",
                "status": "abstained",
                "time": 0,
                "output_summary": "Insufficient evidence – abstaining",
            })
            return self._abstain_result(query, agent_trace, pipeline_start,
                                        verification["reason"],
                                        retrieved_chunks=retrieval["results"],
                                        evidence_confidence=verification["confidence_score"])

        # ── Regeneration Loop ──
        attempt = 0
        best_result = None

        while attempt <= max_attempts:
            # ── Step 4: Generate Answer ──
            gen_result = self.answer_generator.run(
                query, verification["supporting_chunks"],
                verification["evidence_status"], temperature
            )
            agent_trace.append({
                "agent": f"Answer Generator (attempt {attempt + 1})",
                "status": gen_result["status"],
                "time": gen_result["execution_time"],
                "output_summary": f"Generated answer ({len(gen_result['answer'])} chars)",
            })

            if gen_result["status"] == "abstained":
                return self._abstain_result(query, agent_trace, pipeline_start,
                                            "Generator abstained",
                                            retrieved_chunks=retrieval["results"],
                                            evidence_confidence=verification["confidence_score"])

            # ── Step 5: Hallucination Critic ──
            critic_result = self.hallucination_critic.run(
                gen_result["answer"], verification["supporting_chunks"], temperature
            )

            # ── Step 6: Compute Score ──
            score = compute_hallucination_score(
                critic_result, verification["confidence_score"],
                verification["supporting_chunks"]
            )

            agent_trace.append({
                "agent": f"Hallucination Critic (attempt {attempt + 1})",
                "status": critic_result["status"],
                "time": critic_result["execution_time"],
                "output_summary": (
                    f"Risk: {score['hallucination_risk']:.2f}, "
                    f"Support: {score['support_percentage']:.1f}%, "
                    f"Status: {score['verification_status']}"
                ),
            })

            best_result = {
                "query": query,
                "answer": gen_result["answer"],
                "sources": gen_result["sources_used"],
                "retrieved_chunks": retrieval["results"],
                "supporting_chunks": verification["supporting_chunks"],
                "rejected_chunks": verification["rejected_chunks"],
                "evidence_status": verification["evidence_status"],
                "evidence_confidence": verification["confidence_score"],
                "claims": critic_result.get("claims", []),
                "hallucination_risk": score["hallucination_risk"],
                "support_percentage": score["support_percentage"],
                "verification_status": score["verification_status"],
                "score_details": score,
                "critic_explanation": critic_result.get("explanation", ""),
                "pipeline": "multi_agent_rag",
                "attempt": attempt + 1,
                "agent_trace": agent_trace,
            }

            # Check if answer passes
            if score["hallucination_risk"] <= hallucination_threshold:
                break

            # If too risky and we have attempts left, try again
            attempt += 1
            if attempt <= max_attempts:
                agent_trace.append({
                    "agent": "Decision: Regenerate",
                    "status": "regenerating",
                    "time": 0,
                    "output_summary": (
                        f"Risk {score['hallucination_risk']:.2f} > threshold "
                        f"{hallucination_threshold:.2f} – regenerating"
                    ),
                })
                # Increase temperature slightly for diversity
                temperature = min(temperature + 0.1, 0.8)

        # Final decision
        if best_result["hallucination_risk"] > hallucination_threshold:
            # Even after retries the risk is too high
            if best_result["hallucination_risk"] > 0.7:
                best_result["answer"] = (
                    "Insufficient evidence in the knowledge base to provide a reliable answer. "
                    "(Answer was generated but failed verification.)"
                )
                best_result["verification_status"] = "REJECTED"

        agent_trace.append({
            "agent": "Final Decision",
            "status": best_result["verification_status"],
            "time": 0,
            "output_summary": f"Final status: {best_result['verification_status']}",
        })

        best_result["processing_time"] = round(time.time() - pipeline_start, 3)
        best_result["agent_trace"] = agent_trace
        return best_result

    # ── Helpers ──
    def _abstain_result(self, query, agent_trace, pipeline_start, reason,
                        retrieved_chunks=None, evidence_confidence=0.0):
        return {
            "query": query,
            "answer": "Insufficient evidence in the knowledge base to provide a reliable answer.",
            "sources": [],
            "retrieved_chunks": retrieved_chunks or [],
            "supporting_chunks": [],
            "rejected_chunks": [],
            "evidence_status": "insufficient",
            "evidence_confidence": evidence_confidence,
            "claims": [],
            "hallucination_risk": 0.0,
            "support_percentage": 0.0,
            "verification_status": "INSUFFICIENT_EVIDENCE",
            "score_details": {},
            "critic_explanation": reason,
            "pipeline": "multi_agent_rag",
            "attempt": 0,
            "agent_trace": agent_trace,
            "processing_time": round(time.time() - pipeline_start, 3),
        }
