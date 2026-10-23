"""BNS Legal Intelligence: retrieval and multi-agent answer pipeline."""
import os, re, json, time
from pathlib import Path
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
GEMINI_KEY=os.getenv("GEMINI_API_KEY","")
ROOT=Path(__file__).resolve().parent
_EMBEDDER=None

def get_embedding_model():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER

def _chunks_from_sources():
    docs=[]
    data_dir=ROOT/"data"
    for p in sorted(data_dir.glob("*.txt")):
        docs.append((p.stem,p.read_text(encoding="utf-8",errors="ignore")))
    pdf=ROOT/"Final_BNS.pdf"
    if pdf.exists():
        try:
            from pypdf import PdfReader
            text="\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
            if text.strip(): docs.insert(0,("Final_BNS",text))
        except Exception as exc:
            print(f"[Index] Could not extract PDF text: {exc}")
    chunks=[]
    for name,text in docs:
        text=re.sub(r"\s+"," ",text).strip()
        size,overlap=900,120
        for start in range(0,len(text),size-overlap):
            part=text[start:start+size].strip()
            if len(part)>40:
                chunks.append({"chunk_id":len(chunks),"document_name":name,"text":part})
    return chunks

class VectorStore:
    def __init__(self,index_dir="faiss_index"):
        self.index_dir=ROOT/index_dir
        self.index=None
        self.chunks=[]
    def load(self):
        ip,mp=self.index_dir/"index.faiss",self.index_dir/"metadata.json"
        if ip.exists() and mp.exists():
            self.index=faiss.read_index(str(ip))
            self.chunks=json.loads(mp.read_text(encoding="utf-8"))
            return True
        return self.build()
    def build(self):
        self.chunks=_chunks_from_sources()
        if not self.chunks:
            print("[Index] No text could be extracted. Check Final_BNS.pdf and data/*.txt.")
            return False
        print(f"[Index] Building embeddings for {len(self.chunks)} text chunks (first run may download the model)...")
        model=get_embedding_model()
        vectors=model.encode([c["text"] for c in self.chunks],convert_to_numpy=True,show_progress_bar=True).astype("float32")
        self.index=faiss.IndexFlatL2(vectors.shape[1]); self.index.add(vectors)
        self.index_dir.mkdir(parents=True,exist_ok=True)
        faiss.write_index(self.index,str(self.index_dir/"index.faiss"))
        (self.index_dir/"metadata.json").write_text(json.dumps(self.chunks,ensure_ascii=False),encoding="utf-8")
        print(f"[Index] Saved {len(self.chunks)} vectors.")
        return True
    def search(self,query_emb,top_k=5):
        if self.index is None or not self.index.ntotal: return []
        dists,ids=self.index.search(query_emb.reshape(1,-1).astype("float32"),min(top_k,self.index.ntotal))
        return [dict(self.chunks[i],similarity_score=round(1/(1+float(d)),4)) for d,i in zip(dists[0],ids[0]) if i>=0]
    @property
    def num_vectors(self): return self.index.ntotal if self.index is not None else 0
    @property
    def num_chunks(self): return len(self.chunks)

class QueryAnalyzerAgent:
    def run(self,q):
        t=time.time(); low=q.lower(); m=re.search(r"(?:section|sec\.?|s\.)\s*(\d{1,3})\b",low)
        subs=[q]
        if m: subs.append(f"Section {m.group(1)} BNS")
        for key,extra in [("murder","Section 103 punishment for murder"),("theft","Section 303 theft movable property"),("organized crime","Section 111 organized crime"),("private defence","Sections 34 35 right of private defence"),("self defence","Sections 34 35 right of private defence")]:
            if key in low: subs.append(extra)
        intent="definition" if any(x in low for x in ["what is","define","constitutes"]) else "punishment" if any(x in low for x in ["punish","penalty"]) else "legal_inquiry"
        return {"intent":intent,"target_section":m.group(1) if m else None,"sub_queries":list(dict.fromkeys(subs)),"time":round(time.time()-t,3)}

class RetrieverAgent:
    def __init__(self,vs): self.vs=vs
    def run(self,q,sub_queries=None,top_k=5):
        t=time.time(); model=get_embedding_model(); results=[]; seen=set()
        for query in (sub_queries or [q]):
            emb=model.encode([query],convert_to_numpy=True)[0]
            for hit in self.vs.search(emb,top_k):
                if hit["chunk_id"] not in seen: seen.add(hit["chunk_id"]); results.append(hit)
        m=re.search(r"(?:section|sec\.?|s\.)\s*(\d{1,3})\b",q.lower())
        target=m.group(1) if m else None
        if target:
            for c in self.vs.chunks:
                if c["chunk_id"] not in seen and (f"section {target}" in c["text"].lower() or re.search(rf"\b{target}\.\s",c["text"])):
                    results.append(dict(c,similarity_score=.95)); seen.add(c["chunk_id"])
        results.sort(key=lambda x:x["similarity_score"],reverse=True)
        return {"results":results[:top_k],"time":round(time.time()-t,3)}

class EvidenceVerifierAgent:
    def run(self,chunks,threshold=.35):
        t=time.time(); support=[c for c in chunks if c.get("similarity_score",0)>=threshold]
        rejected=[c for c in chunks if c.get("similarity_score",0)<threshold]
        avg=sum(c["similarity_score"] for c in support)/len(support) if support else 0
        confidence=round(min(avg*1.3,1),4) if support else 0
        status="sufficient" if len(support)>=2 and confidence>=.4 else "weak" if support else "insufficient"
        return {"status":status,"confidence":confidence,"supporting":support,"rejected":rejected,"time":round(time.time()-t,3)}

BNS_STATUTE={
"103":("Section 103 (Murder)",["Section 103 concerns punishment for murder. Check the statutory text and applicable subsection for exact wording and exceptions."]),
"303":("Section 303 (Theft)",["Section 303 concerns theft. Check the statutory text and applicable subsection for exact wording and punishment."]),
"106":("Section 106 (Causing death by negligence)",["Section 106 concerns causing death by negligence. Verify the applicable subsection and current statutory wording."]),
"111":("Section 111 (Organized crime)",["Section 111 concerns organized crime. Verify the statutory definition and applicable punishment in the official text."]),
"defence":("Right of private defence",["Sections 34–44 address the right of private defence. Read the relevant sections and limitations in the official text."])}

class AnswerGeneratorAgent:
    def run(self,q,chunks,evidence_status,temperature=.3):
        t=time.time()
        if evidence_status=="insufficient" or not chunks:
            return {"answer":"Insufficient evidence to provide a reliable answer from the indexed sources.","sources":[],"status":"abstained","time":round(time.time()-t,3)}
        sources=sorted({c["document_name"] for c in chunks})
        if GEMINI_KEY and GEMINI_KEY!="your_api_key_here":
            try:
                from google import genai
                from google.genai import types
                client=genai.Client(api_key=GEMINI_KEY)
                ev="\n\n".join(f"[{c['document_name']}] {c['text']}" for c in chunks)
                prompt=("Answer the user's question using only the evidence below. Clearly state uncertainty, "
                        "do not invent legal provisions, and cite the source names. This is not legal advice.\n"
                        f"Question: {q}\nEvidence:\n{ev}\nAnswer:")
                response=client.models.generate_content(model="gemini-2.0-flash",contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature,max_output_tokens=1000))
                if response.text and response.text.strip():
                    return {"answer":response.text.strip(),"sources":sources,"status":"completed","time":round(time.time()-t,3)}
            except Exception as exc: print(f"[Generator] Gemini unavailable; using retrieved evidence: {exc}")
        low=q.lower(); m=re.search(r"(?:section|sec\.?|s\.)\s*(\d{1,3})\b",low)
        key=m.group(1) if m and m.group(1) in BNS_STATUTE else None
        if not key and any(x in low for x in ["private defence","self defence"]): key="defence"
        if key:
            title,points=BNS_STATUTE[key]
            answer=f"**{title} — BNS, 2023**\n\n"+"\n".join("• "+p for p in points)+"\n\n*Verify the exact provision against the official Act.*"
        else:
            lines=[f"**Evidence-based response**\n\nQuestion: {q}"]
            for c in chunks[:3]:
                sentence=re.sub(r"\s+"," ",c["text"]).strip()[:550]
                lines.append(f"\n\n• {sentence} *(Source: {c['document_name']})*")
            answer="".join(lines)
        return {"answer":answer,"sources":sources,"status":"completed","time":round(time.time()-t,3)}

class HallucinationCriticAgent:
    STOPS={"the","a","an","is","are","was","were","in","of","to","and","for","that","it","on","with","as","by","this","from","or"}
    def run(self,answer,chunks,evidence_conf):
        t=time.time()
        if not answer or answer.startswith("Insufficient evidence"):
            return {"claims":[],"hallucination_risk":0.0,"support_percentage":0.0,"verification_status":"ABSTAINED","time":round(time.time()-t,3)}
        evidence=set(re.sub(r"[^a-z0-9\s]"," "," ".join(c["text"] for c in chunks).lower()).split())
        sentences=[s.strip() for s in re.split(r"[.!?\n]",answer) if len(s.strip())>20 and not s.strip().startswith("**")]
        claims=[]; supported=partial=unsupported=0
        for sentence in sentences:
            words=set(re.sub(r"[^a-z0-9\s]"," ",sentence.lower()).split())-self.STOPS
            if len(words)<2: continue
            overlap=len(words & evidence)/len(words)
            status="SUPPORTED" if overlap>=.4 else "PARTIALLY_SUPPORTED" if overlap>=.18 else "UNSUPPORTED"
            if status=="SUPPORTED": supported+=1
            elif status=="PARTIALLY_SUPPORTED": partial+=1
            else: unsupported+=1
            claims.append({"claim":sentence,"status":status,"reason":f"{overlap:.0%} lexical overlap"})
        total=supported+partial+unsupported
        support=(supported+.5*partial)/total if total else 1.0
        retrieval=sum(c.get("similarity_score",0) for c in chunks)/len(chunks) if chunks else 0
        risk=round(max(0,min(1,1-(.5*support+.25*evidence_conf+.25*retrieval))),4)
        verdict="VERIFIED" if risk<=.35 else "MEDIUM_RISK" if risk<=.6 else "HIGH_RISK"
        return {"claims":claims,"hallucination_risk":risk,"support_percentage":round(support*100,1),"verification_status":verdict,"time":round(time.time()-t,3)}

class RAGOrchestrator:
    def __init__(self,vs):
        self.vs=vs; self.analyzer=QueryAnalyzerAgent(); self.retriever=RetrieverAgent(vs)
        self.verifier=EvidenceVerifierAgent(); self.generator=AnswerGeneratorAgent(); self.critic=HallucinationCriticAgent()
    def run_multi_agent_rag(self,query,top_k=5,temperature=.3,hallucination_threshold=.4,evidence_threshold=.35,max_attempts=2):
        t=time.time(); trace=[]; ana=self.analyzer.run(query)
        trace.append({"agent":"Query Analyzer","status":"completed","time":ana["time"],"output_summary":f"Intent: {ana['intent']}; {len(ana['sub_queries'])} sub-queries"})
        ret=self.retriever.run(query,ana["sub_queries"],top_k)
        trace.append({"agent":"Retriever","status":"completed","time":ret["time"],"output_summary":f"Retrieved {len(ret['results'])} chunks"})
        ver=self.verifier.run(ret["results"],evidence_threshold)
        trace.append({"agent":"Evidence Verifier","status":ver["status"],"time":ver["time"],"output_summary":f"Evidence confidence: {ver['confidence']:.2f}"})
        if ver["status"]=="insufficient":
            return {"query":query,"answer":"Insufficient evidence.","pipeline":"multi_agent_rag","verification_status":"INSUFFICIENT_EVIDENCE","hallucination_risk":0.0,"support_percentage":0.0,"evidence_confidence":0.0,"retrieved_chunks":ret["results"],"supporting_chunks":[],"claims":[],"agent_trace":trace,"processing_time":round(time.time()-t,3)}
        for attempt in range(max_attempts+1):
            gen=self.generator.run(query,ver["supporting"],ver["status"],temperature)
            trace.append({"agent":f"Answer Generator (attempt {attempt+1})","status":gen["status"],"time":gen["time"],"output_summary":f"Generated {len(gen['answer'])} chars"})
            crit=self.critic.run(gen["answer"],ver["supporting"],ver["confidence"])
            trace.append({"agent":f"Hallucination Critic (attempt {attempt+1})","status":"completed","time":crit["time"],"output_summary":f"Estimated risk: {crit['hallucination_risk']:.2f}"})
            if crit["hallucination_risk"]<=hallucination_threshold or attempt==max_attempts:
                return {"query":query,"answer":gen["answer"],"sources":gen["sources"],"pipeline":"multi_agent_rag","retrieved_chunks":ret["results"],"supporting_chunks":ver["supporting"],"evidence_confidence":ver["confidence"],"hallucination_risk":crit["hallucination_risk"],"support_percentage":crit["support_percentage"],"verification_status":crit["verification_status"],"claims":crit["claims"],"agent_trace":trace,"processing_time":round(time.time()-t,3)}
    def run_basic_rag(self,query,top_k=5,temperature=.3):
        t=time.time(); ret=self.retriever.run(query,[query],top_k)
        gen=self.generator.run(query,ret["results"],"sufficient",temperature)
        avg=sum(c["similarity_score"] for c in ret["results"])/len(ret["results"]) if ret["results"] else 0
        return {"query":query,"answer":gen["answer"],"sources":gen["sources"],"pipeline":"basic_rag","retrieved_chunks":ret["results"],"supporting_chunks":ret["results"],"evidence_confidence":round(avg,4),"hallucination_risk":None,"support_percentage":None,"verification_status":"NOT_VERIFIED","claims":[],"agent_trace":[{"agent":"Retriever","status":"completed","time":ret["time"]},{"agent":"Answer Generator","status":gen["status"],"time":gen["time"]}],"processing_time":round(time.time()-t,3)}

if __name__=="__main__":
    store=VectorStore()
    if not store.load(): raise SystemExit("Unable to build/load the FAISS index.")
    result=RAGOrchestrator(store).run_multi_agent_rag("What does Section 103 of BNS cover?")
    print(result["verification_status"],result["answer"])
