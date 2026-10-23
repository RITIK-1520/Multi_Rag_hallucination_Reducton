"""FastAPI backend serving the BNS Multi-Agent RAG API and React build."""
import os, time, threading, webbrowser
from pathlib import Path
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from rag_system import VectorStore, RAGOrchestrator, GEMINI_KEY, get_embedding_model

ROOT=Path(__file__).resolve().parent
app=FastAPI(title="BNS Legal Intelligence",version="2.1.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=False,allow_methods=["*"],allow_headers=["*"])
vs=VectorStore()
if not vs.load(): print("[Server] No index available. Check Final_BNS.pdf and data/*.txt.")
orchestrator=RAGOrchestrator(vs)

class QueryReq(BaseModel):
    query:str=Field(min_length=1,max_length=5000)
    pipeline:str="multi_agent"
    top_k:int=Field(default=5,ge=1,le=20)
    temperature:float=Field(default=.3,ge=0,le=1)
    hallucination_threshold:float=Field(default=.4,ge=0,le=1)
    evidence_threshold:float=Field(default=.35,ge=0,le=1)
    max_attempts:int=Field(default=2,ge=0,le=4)

def clean_payload(obj):
    if isinstance(obj,dict): return {k:clean_payload(v) for k,v in obj.items()}
    if isinstance(obj,list): return [clean_payload(v) for v in obj]
    if hasattr(obj,"item"): return obj.item()
    if isinstance(obj,(float,int,str,bool)) or obj is None: return obj
    return str(obj)

@app.get("/api/health")
def health(): return {"status":"ok","service":"BNS Multi-Agent RAG"}

@app.get("/api/stats")
def stats():
    docs=sorted({c["document_name"] for c in vs.chunks})
    live=bool(GEMINI_KEY and GEMINI_KEY!="your_api_key_here")
    return {"status":"online","num_documents":len(docs),"documents":docs,"num_chunks":vs.num_chunks,"num_vectors":vs.num_vectors,"is_live_gemini":live,"mode":"Live Gemini" if live else "Offline Grounded Mode","defaults":{"top_k":5,"hallucination_threshold":.4,"evidence_threshold":.35,"max_attempts":2,"temperature":.3}}

@app.get("/api/questions")
def sample_questions():
    return {"questions":[
      {"id":"q1","question":"What does Section 103 of BNS cover?","category":"Offences Against Life","difficulty":"Medium"},
      {"id":"q2","question":"What does Section 303 of BNS cover?","category":"Offences Against Property","difficulty":"Easy"},
      {"id":"q3","question":"What is the right of private defence under BNS?","category":"General Exceptions","difficulty":"Hard"},
      {"id":"q4","question":"What does Section 111 of BNS address?","category":"Organized Crime","difficulty":"Medium"},
      {"id":"q5","question":"What does Section 106 of BNS address?","category":"Negligence","difficulty":"Medium"}]}

@app.post("/api/query")
def query_endpoint(req:QueryReq):
    q=req.query.strip()
    if not q: raise HTTPException(400,"Query cannot be empty.")
    try:
        if req.pipeline=="basic": result=orchestrator.run_basic_rag(q,req.top_k,req.temperature)
        else: result=orchestrator.run_multi_agent_rag(q,req.top_k,req.temperature,req.hallucination_threshold,req.evidence_threshold,req.max_attempts)
        return clean_payload(result)
    except Exception as exc:
        raise HTTPException(500,f"Error processing query: {exc}") from exc

@app.post("/api/compare")
def compare_endpoint(req:QueryReq):
    q=req.query.strip()
    if not q: raise HTTPException(400,"Query cannot be empty.")
    try:
        multi=orchestrator.run_multi_agent_rag(q,req.top_k,req.temperature,req.hallucination_threshold,req.evidence_threshold,req.max_attempts)
        basic=orchestrator.run_basic_rag(q,req.top_k,req.temperature)
        return clean_payload({"query":q,"multi_agent":multi,"basic":basic})
    except Exception as exc:
        raise HTTPException(500,f"Error processing comparison: {exc}") from exc

dist=ROOT/"frontend"/"dist"
if dist.exists() and (dist/"index.html").exists():
    app.mount("/assets",StaticFiles(directory=str(dist/"assets"),check_dir=False),name="assets")
    @app.get("/{path:path}")
    def serve_ui(path:str):
        if path.startswith("api/"): raise HTTPException(404,"API endpoint not found")
        file=dist/path
        return FileResponse(file if file.is_file() else dist/"index.html",headers={"Cache-Control":"no-cache"})
else:
    @app.get("/",response_class=HTMLResponse)
    def build_frontend_notice():
        return HTMLResponse("<h2>BNS Legal Intelligence API is running</h2><p>Build the React UI with <code>cd frontend && npm install && npm run build</code>.</p><p>API health: <a href='/api/health'>/api/health</a> · API docs: <a href='/docs'>/docs</a></p>")

if __name__=="__main__":
    print("BNS Legal Intelligence: http://127.0.0.1:8000")
    uvicorn.run(app,host="127.0.0.1",port=8000)
