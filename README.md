# BNS Legal Intelligence — Multi-Agent RAG

An academic web application demonstrating evidence-grounded question answering over the Bharatiya Nyaya Sanhita (BNS) and included computer-science text sources. It combines a React/Vite interface, a FastAPI backend, FAISS vector retrieval, and a five-stage agent pipeline.

> **Legal and research disclaimer:** This is a student prototype, not legal advice. The critic uses lexical-overlap heuristics, not legal-entailment verification. Verify legal wording, sections, and penalties against the official Act. Do not treat the displayed confidence or “verified” status as a guarantee.

## Features

- React + Vite frontend and FastAPI JSON API.
- Query Analyzer, Retriever, Evidence Verifier, Answer Generator, and Hallucination Critic.
- FAISS `IndexFlatL2` dense retrieval using `sentence-transformers/all-MiniLM-L6-v2`.
- Basic RAG vs. Multi-Agent RAG comparison.
- Retrieved evidence chunks, source names, agent trace, and heuristic claim-support report.
- Optional Google Gemini generation through `GEMINI_API_KEY`; retrieved-text fallback when no key is configured or the API fails.
- Builds the vector index automatically from the included text files and extractable text in `Final_BNS.pdf` if a prebuilt index is not present.

## Architecture

```text
React + Vite browser UI
          │ HTTP / JSON
          ▼
FastAPI API (app.py)
          │
          ▼
RAGOrchestrator (rag_system.py)
   ├─ Query Analyzer
   ├─ Retriever ──► FAISS vector index
   ├─ Evidence Verifier
   ├─ Answer Generator ──► optional Gemini API
   └─ Hallucination Critic
          │
          ▼
Answer, source chunks, scores, and agent trace
```

## Repository layout

```text
.
├── app.py
├── rag_system.py
├── requirements.txt
├── start.bat
├── .env.example
├── Final_BNS.pdf
├── data/                   # Included CS/AI reference text files
├── frontend/               # React/Vite source and package files
│   └── src/
├── evaluation/             # Example evaluation questions
├── outputs/                # Example reports and result exports
├── docs/                   # Architecture image
└── archive/tests/          # Archived tests from the original project
```

The generated `faiss_index/` directory is intentionally not committed. The application builds it from source documents the first time it starts. The index can be large and is reproducible.

## Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- Node.js and npm, only if building the frontend
- Internet access on first run to install dependencies and download the embedding model
- Optional Google Gemini API key for live generation

## Run on Windows

1. Clone/download the repository and open PowerShell in its root directory.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the backend dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Optional: copy `.env.example` to `.env` and set your Gemini key:

   ```env
   GEMINI_API_KEY=your_actual_api_key
   ```

   Never commit a real API key. The `.env` file is ignored by Git.

5. Build the frontend:

   ```powershell
   cd frontend
   npm install
   npm run build
   cd ..
   ```

6. Start the app:

   ```powershell
   python app.py
   ```

   Or double-click `start.bat`: it creates a Python virtual environment, installs backend dependencies, and builds the React UI if Node.js/npm are available and the build is missing. Open **http://127.0.0.1:8000**. FastAPI API documentation is available at **http://127.0.0.1:8000/docs**.

On the first start, the app reads `Final_BNS.pdf` (when PDF text extraction is available) and `data/*.txt`, creates chunks, downloads/loads the Sentence Transformers embedding model, and writes the local FAISS index and metadata to `faiss_index/`. This initial step may take a few minutes. Subsequent starts reuse the generated index.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | Loaded documents, chunks, vector count, and runtime mode |
| `GET` | `/api/questions` | Example questions |
| `POST` | `/api/query` | Run Basic RAG or Multi-Agent RAG |
| `POST` | `/api/compare` | Run both pipelines for one query |

Example request body:

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

## How the agents work

1. **Query Analyzer** identifies basic intent, detects some section numbers, and adds targeted sub-queries for selected topics.
2. **Retriever** embeds the query/sub-queries and searches the FAISS index. It also applies a simple keyword match for detected section numbers.
3. **Evidence Verifier** separates retrieved chunks by a configurable similarity threshold and estimates evidence confidence.
4. **Answer Generator** uses Gemini when configured and reachable; otherwise it extracts relevant text from retrieved evidence and uses limited predefined topic summaries.
5. **Hallucination Critic** compares answer words with retrieved evidence and calculates a heuristic claim-support and risk estimate. The orchestrator may retry generation up to the configured attempt limit.

### Limitations

- Lexical overlap is not a reliable substitute for legal entailment or human review.
- A retry repeats generation with the same evidence; the critic's specific feedback is not automatically added to the next generation prompt.
- The bundled sources include general computer-science/AI notes as well as the BNS PDF. This is a mixed-source knowledge base, not exclusively a legal corpus.
- PDF extraction can vary by document encoding or whether its pages contain selectable text.
- Offline fallback is evidence-extraction oriented and is not equivalent to a capable general-purpose LLM.

## Troubleshooting

- **Embedding model download fails:** Check network access, then restart. The model is cached locally after the first successful download.
- **PDF text is not indexed:** Confirm that the PDF contains selectable text and that `pypdf` is installed. Scanned pages may require OCR, which is not included.
- **Frontend does not load:** Run `npm install` and `npm run build` from `frontend/`, then restart the backend.
- **Gemini not enabled:** The app can run without a key using its retrieval-based fallback, but answer quality and variety are limited.
- **Index needs rebuilding:** Stop the app, delete the generated `faiss_index/` folder, and start it again.

## License

No license is currently declared. Add a license file before granting reuse or redistribution rights.
