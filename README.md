# BNS Legal Intelligence — Multi-Agent RAG

A local web application that answers questions about the Bharatiya Nyaya Sanhita (BNS) using retrieval-augmented generation (RAG), a FAISS vector index, and a multi-stage verification pipeline.

> **Important:** This is an academic prototype, not a legal-advice service. Its evidence and hallucination scores are heuristic indicators and do not guarantee that a legal answer is correct. Always verify legal claims against an authoritative source.

## Features

- **React + Vite frontend** served by a **FastAPI** backend.
- Five pipeline stages: Query Analyzer, Retriever, Evidence Verifier, Answer Generator, and Hallucination Critic.
- FAISS `IndexFlatL2` search using `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- Basic RAG and Multi-Agent RAG paths, plus a side-by-side comparison view.
- Evidence-chunk viewer, agent execution trace, and heuristic claim-support/risk report.
- Offline fallback using the local index and predefined summaries; optional Google Gemini generation when configured.

## Architecture

```text
Browser (React UI)
        │ HTTP / JSON
        ▼
FastAPI backend (app.py)
        │
        ▼
RAGOrchestrator (rag_system.py)
        ├── Query Analyzer
        ├── Retriever ─────► FAISS index + metadata
        ├── Evidence Verifier
        ├── Answer Generator ─────► Gemini (optional) / offline fallback
        └── Hallucination Critic
        │
        ▼
Answer + retrieved chunks + scores + agent trace
```

## Project structure

```text
.
├── app.py                      # FastAPI API and static React app server
├── rag_system.py               # Vector store and five-agent RAG pipeline
├── requirements.txt            # Python dependencies
├── start.bat                   # Windows launcher
├── .env.example                # Optional API-key template
├── Final_BNS.pdf               # BNS reference document supplied with project
├── faiss_index/
│   ├── index.faiss             # Prebuilt vector index
│   └── metadata.json           # Chunk text and document metadata
├── frontend/
│   ├── src/                    # React UI source
│   └── dist/                   # Built UI served by FastAPI
├── data/                       # Text files included with the project
├── evaluation/                 # Example evaluation questions
├── outputs/                    # Previously generated example reports/results
└── docs/                       # Architecture diagram
```

## Requirements

- Python 3.10+ (Python 3.11 is a reasonable choice)
- Node.js and npm are needed only if you want to rebuild the frontend
- Internet access is needed for initial installation/model downloads and optional Gemini use

## Run on Windows

1. Download or clone this repository and open a terminal in the repository folder.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install Python dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. (Optional) Enable Gemini. Copy `.env.example` to `.env` and set your key:

   ```env
   GEMINI_API_KEY=your_actual_api_key
   ```

   Keep `.env` private. Never commit a real API key.

5. Start the app:

   ```powershell
   python app.py
   ```

   Or double-click `start.bat`. The application opens at **http://127.0.0.1:8000**.

The application expects `faiss_index/index.faiss` and `faiss_index/metadata.json` to exist. The prebuilt index is supplied with the project ZIP. The local embedding model must also be available to Sentence Transformers; the app sets Hugging Face/Transformers offline flags, so model files need to have been downloaded/cached before runtime in an offline environment.

## Run / rebuild the frontend

The compiled frontend in `frontend/dist/` is served by FastAPI, so you do not need Node.js for normal use. To edit and rebuild the React UI:

```powershell
cd frontend
npm install
npm run build
cd ..
python app.py
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Backend health check |
| `GET` | `/api/stats` | Loaded document, chunk and vector statistics; runtime mode |
| `GET` | `/api/questions` | Example legal questions |
| `POST` | `/api/query` | Run Basic RAG or Multi-Agent RAG |
| `POST` | `/api/compare` | Run both pipelines for the same query |

Example request:

```json
{
  "query": "What does Section 103 of BNS cover?",
  "pipeline": "multi_agent",
  "top_k": 5,
  "temperature": 0.3,
  "hallucination_threshold": 0.4,
  "evidence_threshold": 0.35,
  "max_attempts": 2
}
```

## How the five stages work

1. **Query Analyzer:** Uses keyword and regular-expression rules to infer intent and detect some section numbers; adds targeted sub-queries for known topics.
2. **Retriever:** Embeds the query/sub-queries, searches FAISS, removes duplicate chunk IDs, and applies a simple statutory keyword boost for matching section text.
3. **Evidence Verifier:** Splits chunks by the configured score threshold and computes a heuristic evidence-confidence value. It can mark evidence sufficient, weak, or insufficient.
4. **Answer Generator:** Uses Gemini if a key is configured and the API call succeeds. Otherwise, it uses predefined BNS summaries for selected topics or extracts sentences from retrieved chunks.
5. **Hallucination Critic:** Estimates claim support with lexical word overlap against retrieved chunks and calculates a heuristic risk score. The orchestrator may retry until the risk threshold is met or the attempt limit is reached.

**Implementation note:** The retry loop reruns generation and critique with the same supporting chunks; critic feedback is not explicitly fed back into the generation prompt. The lexical-overlap critic is not a legal-entailment checker. Therefore, a `VERIFIED` label is not proof of legal correctness.

## Basic RAG vs Multi-Agent RAG

- **Basic RAG:** Retrieves chunks and calls the generator; the result is marked `NOT_VERIFIED` and does not receive the critic's risk score.
- **Multi-Agent RAG:** Adds query analysis, evidence sufficiency filtering, post-generation claim scoring, and a retry/stop decision.

## Knowledge-base note

The supplied metadata contains **1,388 chunks from nine document names**. The chunk inventory includes `Final_BNS` plus general AI/computer-science reference files. It is therefore a mixed knowledge base, not exclusively BNS law. For a legal demonstration, inspect retrieved chunks and verify the final answer directly against the official BNS text.

## Troubleshooting

- **FAISS index not found:** Check that `faiss_index/index.faiss` and `faiss_index/metadata.json` are present beside `app.py`.
- **Embedding model error while offline:** Ensure `all-MiniLM-L6-v2` is already available in the local Sentence Transformers cache before running with offline flags.
- **Gemini errors:** Check `.env`, the `GEMINI_API_KEY` value, package installation, and API access. The generator attempts its offline fallback if Gemini generation fails.
- **Frontend changes not visible:** Rebuild using `npm run build` inside `frontend/`; then refresh the browser.
- **Do not rely on scores alone:** Validate citations and legal wording against an authoritative legal source.

## License

No license is specified in this repository yet. Add an appropriate license if you intend to grant reuse or redistribution rights.
