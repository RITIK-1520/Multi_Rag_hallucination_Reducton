import re
from src.vector_store import VectorStore
from src.orchestrator import Orchestrator

vs = VectorStore()
vs.load()
orch = Orchestrator(vs)

q = "What are the legal rights of private defence (self-defence) under Bharatiya Nyaya Sanhita?"
ret = orch.retriever.run(q, top_k=6)

for idx, c in enumerate(ret["results"]):
    raw = c["text"]
    # 1. Unstick 2023 followed by section number
    raw = re.sub(r'(?i)THE BHARATIYA NYAYA SANHITA,?\s*2023(\d{1,3})\.', r'\n\n=== SECTION \1: ', raw)
    # 2. Fix Section numbers like "34. THINGS DONE..."
    raw = re.sub(r'(?m)^(\d{1,3})\.\s*([A-Z\s]{4,})', r'\n=== SECTION \1: \2\n', raw)
    # 3. Clean word splits
    raw = re.sub(r'P\s*R\s*I\s*V\s*A\s*T\s*E', 'PRIVATE', raw, flags=re.I)
    raw = re.sub(r'D\s*E\s*F\s*E\s*N\s*C\s*E', 'DEFENCE', raw, flags=re.I)
    print(f"--- CHUNK {idx+1} (score {c['similarity_score']:.2f}) ---")
    print(raw[:600])
    print("=" * 40)
