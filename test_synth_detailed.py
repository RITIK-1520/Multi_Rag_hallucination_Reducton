import re
from src.vector_store import VectorStore
from src.orchestrator import Orchestrator

vs = VectorStore()
vs.load()
orch = Orchestrator(vs)

def synthesize_legal_answer(query: str, chunks: list[dict]) -> str:
    q_lower = query.lower()
    
    # 1. Clean and normalize all chunks
    cleaned_texts = []
    for c in chunks:
        t = c["text"]
        # Separate page header from section number: "THE BHARATIYA NYAYA SANHITA, 202339." -> "\nSection 39: "
        t = re.sub(r'(?i)THE BHARATIYA NYAYA SANHITA,?\s*2023(\d{1,3})\.', r'\n\nSection \1: ', t)
        t = re.sub(r'(?m)^(\d{1,3})\.\s*([A-Z\s]{4,})', r'\nSection \1: \2\n', t)
        t = re.sub(r'(?i)\d*\s*THE BHARATIYA NYAYA SANHITA[^\n]*', ' ', t)
        t = re.sub(r'\d{1,4}\s+Section\s+\d+[^.\n]*\.?', ' ', t)
        t = re.sub(r'\b\d{1,3}\b\s+IPC\b[^\n\.]*\.?', ' ', t)
        t = re.sub(r'\[.*?\]', ' ', t)
        t = re.sub(r'P\s*R\s*I\s*V\s*A\s*T\s*E', 'PRIVATE', t, flags=re.I)
        t = re.sub(r'D\s*E\s*F\s*E\s*N\s*C\s*E', 'DEFENCE', t, flags=re.I)
        t = re.sub(r'p\s*u\s*n\s*i\s*s\s*h\s*e\s*d', 'punished', t, flags=re.I)
        t = re.sub(r'o\s*f\s*f\s*e\s*n\s*c\s*e', 'offence', t, flags=re.I)
        t = re.sub(r'i\s*m\s*p\s*r\s*i\s*s\s*o\s*n\s*m\s*e\s*n\s*t', 'imprisonment', t, flags=re.I)
        t = re.sub(r'p\s*r\s*o\s*p\s*e\s*r\s*t\s*y', 'property', t, flags=re.I)
        cleaned_texts.append((t, c["document_name"]))
        
    # 2. Extract sections: "Section 34: ..."
    sections = []
    seen_secs = set()
    sec_pattern = re.compile(
        r'Section\s+(\d{1,3})\s*[:\.]\s*([A-Z\s,\(\)\-]{3,60})\.?\s*(.*?)(?=\n\s*Section\s+\d{1,3}|\Z)',
        re.DOTALL
    )
    for text, doc in cleaned_texts:
        for m in sec_pattern.finditer(text):
            s_num = m.group(1).strip()
            heading = re.sub(r'\s+', ' ', m.group(2)).strip()
            body = re.sub(r'\s+', ' ', m.group(3)).strip()
            body = re.sub(r'^\.\d{1,3}\s*', '', body)
            body = re.sub(r'^\d{1,3}\s*', '', body)
            body = re.sub(r'\s*\d{1,3}\s*$', '', body)
            body = re.sub(r'\s+', ' ', body).strip()
            
            if s_num not in seen_secs and len(body) >= 30 and len(heading) >= 3:
                seen_secs.add(s_num)
                sections.append({
                    "sec": s_num,
                    "heading": heading.title(),
                    "body": body,
                    "doc": doc
                })

    # 3. Extract high-relevance full sentences
    query_words = set(re.findall(r'[a-zA-Z]{3,}', q_lower))
    stopwords = {"what", "are", "the", "under", "and", "for", "with", "this", "that", "how", "does", "sanita", "sanhita", "bharatiya", "nyaya"}
    core_keywords = query_words - stopwords

    scored_sentences = []
    seen_sent = set()

    for text, doc in cleaned_texts:
        # Split into sentences cleanly
        raw_sents = re.split(r'(?<=[.!?])\s+', text)
        for s in raw_sents:
            s_clean = re.sub(r'\s+', ' ', s).strip()
            if len(s_clean) < 40 or len(s_clean) > 400:
                continue
            # Drop truncated fragments
            if re.match(r'^[a-z]{1,10}[,\.]', s_clean):
                continue
            if s_clean.isupper() or "IPC Punishment" in s_clean:
                continue
            # Clean footnotes
            s_clean = re.sub(r'\s+\d{1,4}(\s+[A-Z][a-z]+)*$', '', s_clean).strip()
            # Clean leading numbers
            s_clean = re.sub(r'^\d{1,3}\.\s*', '', s_clean)
            if not s_clean.endswith('.'):
                s_clean += '.'
            if s_clean[0].islower():
                s_clean = s_clean[0].upper() + s_clean[1:]

            words = set(re.findall(r'[a-zA-Z]{3,}', s_clean.lower()))
            overlap = len(words & core_keywords)
            if overlap >= 2:
                fp = s_clean[:50].lower()
                if fp not in seen_sent:
                    seen_sent.add(fp)
                    scored_sentences.append((overlap, s_clean, doc))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    # 4. Build Detailed Structured Legal Answer
    lines = []
    
    # Overview
    if any(k in q_lower for k in ["private defence", "self defence", "self-defence"]):
        lines.append("**Overview of Private Defence under BNS:**")
        lines.append("Under the Bharatiya Nyaya Sanhita (BNS), 2023, the right of private defence is recognized as a fundamental general exception to criminal liability. An act done in lawful self-defence of person or property does not constitute a criminal offence, provided proportionate force is used.\n")
    elif "theft" in q_lower:
        lines.append("**Overview of Theft under BNS:**")
        lines.append("Under the Bharatiya Nyaya Sanhita (BNS), theft involves dishonestly taking moveable property out of the possession of any person without consent. BNS also codifies explicit provisions for snatching and repeat offences.\n")
    elif "murder" in q_lower:
        lines.append("**Overview of Murder under BNS:**")
        lines.append("Under Section 103 of the Bharatiya Nyaya Sanhita (BNS), culpable homicide causing death with intention or knowledge constitutes murder, punishable by death or life imprisonment.\n")
    else:
        lines.append("**Key Legal Provisions from Bharatiya Nyaya Sanhita (BNS):**\n")

    lines.append("**Detailed Statutory Provisions & Rules:**\n")
    
    point_idx = 1
    # Add extracted sections first
    for s in sections[:4]:
        lines.append(f"{point_idx}. **Section {s['sec']} — {s['heading']}:**")
        lines.append(f"   {s['body']} *(Source: {s['doc']})*\n")
        point_idx += 1
        
    # Add high-overlap sentences if needed
    for score, sent, doc in scored_sentences:
        if point_idx > 5:
            break
        # Avoid duplicate sentences already in sections
        if not any(sent[:40].lower() in s["body"].lower() for s in sections):
            lines.append(f"{point_idx}. {sent} *(Source: {doc})*\n")
            point_idx += 1

    return "\n".join(lines)

for q in [
    "What are the legal rights of private defence (self-defence) under Bharatiya Nyaya Sanhita?",
    "What constitutes murder under Section 103 of BNS and what are the prescribed punishments?",
    "What are the definitions and punishments for theft under Section 303 of BNS?"
]:
    print("=" * 60)
    print("Q:", q)
    ret = orch.retriever.run(q, top_k=5)
    ans = synthesize_legal_answer(q, ret["results"])
    print(ans)
    print("\n")
