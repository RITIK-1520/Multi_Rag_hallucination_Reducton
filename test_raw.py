import re
from src.vector_store import VectorStore
from src.orchestrator import Orchestrator

vs = VectorStore()
vs.load()
orch = Orchestrator(vs)

q = "What are the legal rights of private defence (self-defence) under Bharatiya Nyaya Sanhita?"
ret = orch.retriever.run(q, top_k=5)

for i in [0, 1, 3]:
    raw = ret['results'][i]['text']
    print(f"=== RAW CHUNK {i+1} ===")
    print(repr(raw[:300]))
