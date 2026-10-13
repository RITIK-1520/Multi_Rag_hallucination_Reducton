"""
server.py – FastAPI Backend for BNS Legal Intelligence Multi-Agent RAG.

Provides REST endpoints for querying the BNS knowledge base, executing both
Multi-Agent RAG and Basic RAG pipelines, returning agent traces, evidence chunks,
hallucination critics, and system diagnostics.
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    is_api_available, DEFAULT_TOP_K, MAX_TOP_K,
    DEFAULT_HALLUCINATION_THRESHOLD, DEFAULT_EVIDENCE_THRESHOLD,
    DEFAULT_MAX_REGENERATION_ATTEMPTS, DEFAULT_TEMPERATURE,
    MAX_TEMPERATURE,
)
from src.vector_store import VectorStore
from src.orchestrator import Orchestrator
from src.data_loader import load_documents
from src.chunking import chunk_all_documents
from src.embeddings import generate_embeddings
from src.evaluation import load_evaluation_questions

app = FastAPI(
    title="BNS Legal Intelligence API",
    description="Multi-Agent RAG with Zero-Hallucination Guard for Bharatiya Nyaya Sanhita",
    version="2.0.0",
)

# Enable CORS for React frontend (Vite default is 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize VectorStore and Orchestrator
print("[Server] Initializing BNS VectorStore...")
vs = VectorStore()
if not vs.load():
    print("[Server] No existing index found, loading documents and building FAISS index...")
    docs = load_documents()
    if docs:
        chunks = chunk_all_documents(docs)
        texts = [c["text"] for c in chunks]
        embs = generate_embeddings(texts)
        vs.build_index(chunks, embs)
        vs.save()
        print(f"[Server] Built and saved index with {vs.num_vectors} vectors.")
    else:
        print("[Server] WARNING: No documents found to index.")
else:
    print(f"[Server] Loaded vector store with {vs.num_vectors} vectors.")

orchestrator = Orchestrator(vs)


# ═══════════════════════════════════════════════════
# Request / Response Schemas
# ═══════════════════════════════════════════════════
class QueryRequest(BaseModel):
    query: str
    pipeline: str = Field(default="multi_agent", description="'multi_agent' or 'basic'")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=MAX_TEMPERATURE)
    hallucination_threshold: float = Field(default=DEFAULT_HALLUCINATION_THRESHOLD, ge=0.1, le=1.0)
    evidence_threshold: float = Field(default=DEFAULT_EVIDENCE_THRESHOLD, ge=0.0, le=1.0)
    max_attempts: int = Field(default=DEFAULT_MAX_REGENERATION_ATTEMPTS, ge=0, le=5)


class CompareRequest(BaseModel):
    query: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=MAX_TEMPERATURE)
    hallucination_threshold: float = Field(default=DEFAULT_HALLUCINATION_THRESHOLD, ge=0.1, le=1.0)
    evidence_threshold: float = Field(default=DEFAULT_EVIDENCE_THRESHOLD, ge=0.0, le=1.0)
    max_attempts: int = Field(default=DEFAULT_MAX_REGENERATION_ATTEMPTS, ge=0, le=5)


# ═══════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════
@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "BNS Multi-Agent RAG"}


@app.get("/api/stats")
def get_stats():
    doc_names = sorted(list(set(c["document_name"] for c in vs.chunks))) if vs.chunks else []
    return {
        "status": "online",
        "num_documents": len(doc_names),
        "documents": doc_names,
        "num_chunks": vs.num_chunks,
        "num_vectors": vs.num_vectors,
        "is_live_gemini": is_api_available(),
        "mode": "Live Gemini" if is_api_available() else "Offline Deterministic Mode",
        "defaults": {
            "top_k": DEFAULT_TOP_K,
            "hallucination_threshold": DEFAULT_HALLUCINATION_THRESHOLD,
            "evidence_threshold": DEFAULT_EVIDENCE_THRESHOLD,
            "max_attempts": DEFAULT_MAX_REGENERATION_ATTEMPTS,
            "temperature": DEFAULT_TEMPERATURE,
        },
    }


@app.post("/api/query")
def run_query(req: QueryRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        if req.pipeline == "basic":
            res = orchestrator.run_basic_rag(
                query=q,
                top_k=req.top_k,
                temperature=req.temperature,
            )
        else:
            res = orchestrator.run_multi_agent_rag(
                query=q,
                top_k=req.top_k,
                temperature=req.temperature,
                hallucination_threshold=req.hallucination_threshold,
                evidence_threshold=req.evidence_threshold,
                max_attempts=req.max_attempts,
            )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare")
def compare_pipelines(req: CompareRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        multi_res = orchestrator.run_multi_agent_rag(
            query=q,
            top_k=req.top_k,
            temperature=req.temperature,
            hallucination_threshold=req.hallucination_threshold,
            evidence_threshold=req.evidence_threshold,
            max_attempts=req.max_attempts,
        )
        basic_res = orchestrator.run_basic_rag(
            query=q,
            top_k=req.top_k,
            temperature=req.temperature,
        )
        return {
            "query": q,
            "multi_agent": multi_res,
            "basic": basic_res,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/questions")
def get_sample_questions():
    try:
        qs = load_evaluation_questions()
        return {"questions": qs}
    except Exception:
        # Fallback preset sample questions
        return {
            "questions": [
                {
                    "id": "q1",
                    "question": "What constitutes murder under Section 103 of BNS and what are the prescribed punishments?",
                    "category": "Offences Against Life",
                    "difficulty": "Medium",
                },
                {
                    "id": "q2",
                    "question": "What are the definitions and punishments for theft under Section 303 of BNS?",
                    "category": "Offences Against Property",
                    "difficulty": "Easy",
                },
                {
                    "id": "q3",
                    "question": "What are the legal rights of private defence (self-defence) under Bharatiya Nyaya Sanhita?",
                    "category": "General Exceptions",
                    "difficulty": "Hard",
                },
                {
                    "id": "q4",
                    "question": "How does BNS address organized crime and electronic offences under Section 111?",
                    "category": "Organized Crime",
                    "difficulty": "Medium",
                },
                {
                    "id": "q5",
                    "question": "What are the provisions related to causing death by negligence under Section 106?",
                    "category": "Negligence",
                    "difficulty": "Medium",
                },
            ]
        }

# Mount React static build if frontend/dist exists
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        # Don't intercept API routes
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="API route not found")
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
