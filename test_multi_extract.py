import re
from src.vector_store import VectorStore
from src.orchestrator import Orchestrator

vs = VectorStore()
vs.load()
orch = Orchestrator(vs)

test_queries = [
    "What are the legal rights of private defence (self-defence) under Bharatiya Nyaya Sanhita?",
    "What constitutes murder under Section 103 of BNS and what are the prescribed punishments?",
    "What are the definitions and punishments for theft under Section 303 of BNS?",
    "What are the provisions related to causing death by negligence under Section 106?",
    "How does BNS address organized crime and electronic offences under Section 111?"
]

def extract_legal_sections(chunks):
    sections = []
    seen = set()
    
    for c in chunks:
        raw = c["text"]
        
        # 1. Unstick 2023 concatenated with section number
        raw = re.sub(r'(?i)THE BHARATIYA NYAYA SANHITA,?\s*2023(\d{1,3})\.', r'\n\nSECTION_HEADER \1: ', raw)
        # 2. Section at start of lines
        raw = re.sub(r'(?m)^(\d{1,3})\.\s*([A-Z\s]{4,})', r'\nSECTION_HEADER \1: \2\n', raw)
        raw = re.sub(r'P\s*R\s*I\s*V\s*A\s*T\s*E', 'PRIVATE', raw, flags=re.I)
        raw = re.sub(r'D\s*E\s*F\s*E\s*N\s*C\s*E', 'DEFENCE', raw, flags=re.I)
        raw = re.sub(r'p\s*u\s*n\s*i\s*s\s*h\s*e\s*d', 'punished', raw, flags=re.I)
        raw = re.sub(r'o\s*f\s*f\s*e\s*n\s*c\s*e', 'offence', raw, flags=re.I)
        raw = re.sub(r'i\s*m\s*p\s*r\s*i\s*s\s*o\s*n\s*m\s*e\s*n\s*t', 'imprisonment', raw, flags=re.I)
        
        # Split on SECTION_HEADER
        parts = re.split(r'SECTION_HEADER\s*(\d{1,3}):\s*', raw)
        for i in range(1, len(parts), 2):
            sec_num = parts[i].strip()
            content = parts[i+1].strip()
            
            # Extract heading and body
            lines = content.split('\n')
            heading = ""
            body_lines = []
            
            for line in lines:
                l_strip = line.strip()
                # Clean footnote numbers
                l_strip = re.sub(r'\.\d{1,3}$', '.', l_strip)
                l_strip = re.sub(r'^\d{1,3}\s+', '', l_strip)
                if not heading and re.match(r'^[A-Z\s,();:\-]{3,}$', l_strip):
                    heading = l_strip
                else:
                    body_lines.append(l_strip)
                    
            body = " ".join(body_lines)
            body = re.sub(r'\s+', ' ', body).strip()
            
            # Clean IPC cross references from body
            body = re.sub(r'\d{1,4}\s+Section\s+\d+[^.\n]*\.?', '', body)
            body = re.sub(r'\[.*?\]', '', body)
            body = re.sub(r'\s+', ' ', body).strip()
            
            if sec_num not in seen and len(body) > 20:
                seen.add(sec_num)
                sections.append({
                    "sec": sec_num,
                    "heading": heading.title() if heading else f"Section {sec_num}",
                    "body": body[:400]
                })
                
    return sections

for q in test_queries:
    print(f"\n==========================================")
    print(f"QUERY: {q}")
    ret = orch.retriever.run(q, top_k=5)
    secs = extract_legal_sections(ret["results"])
    print(f"Extracted Sections: {len(secs)}")
    for s in secs:
        print(f"  * Section {s['sec']} ({s['heading']}): {s['body'][:150]}...")
