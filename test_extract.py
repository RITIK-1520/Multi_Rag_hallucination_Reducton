import re
from src.vector_store import VectorStore
from src.orchestrator import Orchestrator

vs = VectorStore()
vs.load()
orch = Orchestrator(vs)

q = "What are the legal rights of private defence (self-defence) under Bharatiya Nyaya Sanhita?"
ret = orch.retriever.run(q, top_k=6)

def clean_and_extract(chunks, query):
    all_sections = []
    
    for c in chunks:
        raw = c["text"]
        
        # 1. Unstick 2023 followed by section number
        raw = re.sub(r'(?i)THE BHARATIYA NYAYA SANHITA,?\s*2023(\d{1,3})\.', r'\n\nSection \1. ', raw)
        # Remove remaining headers
        raw = re.sub(r'(?i)\d*\s*THE BHARATIYA NYAYA SANHITA[^\n]*', ' ', raw)
        # Remove footnote markers like 95 Section 103, IPC.
        raw = re.sub(r'\d{1,4}\s+Section\s+\d+[^.\n]*\.?', ' ', raw)
        raw = re.sub(r'\b\d{1,3}\b\s+IPC\b[^\n\.]*\.?', ' ', raw)
        # Fix broken word splits
        raw = re.sub(r'P\s*R\s*I\s*V\s*A\s*T\s*E', 'PRIVATE', raw, flags=re.I)
        raw = re.sub(r'D\s*E\s*F\s*E\s*N\s*C\s*E', 'DEFENCE', raw, flags=re.I)
        raw = re.sub(r'p\s*u\s*n\s*i\s*s\s*h\s*e\s*d', 'punished', raw, flags=re.I)
        raw = re.sub(r'o\s*f\s*f\s*e\s*n\s*c\s*e', 'offence', raw, flags=re.I)
        raw = re.sub(r'i\s*m\s*p\s*r\s*i\s*s\s*o\s*n\s*m\s*e\s*n\s*t', 'imprisonment', raw, flags=re.I)
        raw = re.sub(r'p\s*r\s*o\s*p\s*e\s*r\s*t\s*y', 'property', raw, flags=re.I)
        raw = re.sub(r'\[.*?\]', ' ', raw)
        
        # 2. Extract sections
        sec_pattern = re.compile(
            r'(?:Section\s+)?(\b\d{1,3}\b)\s*\.\s*([A-Z\s,\(\)\-]{3,60})\.?\s*(.*?)(?=(?:Section\s+)?\b\d{1,3}\b\s*\.\s*[A-Z\s]{4,60}|\Z)',
            re.DOTALL
        )
        for m in sec_pattern.finditer(raw):
            sec_num = m.group(1).strip()
            heading = re.sub(r'\s+', ' ', m.group(2)).strip()
            body = re.sub(r'\s+', ' ', m.group(3)).strip()
            # Clean footnotes at start / end of body
            body = re.sub(r'^\d{1,3}\s*', '', body)
            body = re.sub(r'\s*\d{1,3}\s*$', '', body)
            
            # Check if this heading or body is meaningful
            if len(body) > 20 and len(heading) > 3 and not heading.isdigit():
                all_sections.append({
                    "sec": sec_num,
                    "heading": heading,
                    "body": body,
                    "doc": c["document_name"]
                })
                
    return all_sections

sections = clean_and_extract(ret["results"], q)
print(f"Total sections extracted: {len(sections)}")
for s in sections:
    print(f"\n★ Section {s['sec']}: {s['heading'].title()}")
    print(f"  {s['body'][:250]}...")
