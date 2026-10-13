import re
from src.vector_store import VectorStore
from src.orchestrator import Orchestrator

vs = VectorStore()
vs.load()
orch = Orchestrator(vs)

q = "What are the legal rights of private defence (self-defence) under Bharatiya Nyaya Sanhita?"
sub_queries = [q, "34. THINGS DONE IN PRIVATE DEFENCE", "Right of private defence of property and body"]
res = orch.retriever.run(q, sub_queries=sub_queries, top_k=6)
for i, r in enumerate(res['results']):
    text_snippet = re.sub(r'\s+', ' ', r['text'])
    print(f"Chunk {i+1} score {r['similarity_score']:.2f}: {text_snippet[:160]}...")
