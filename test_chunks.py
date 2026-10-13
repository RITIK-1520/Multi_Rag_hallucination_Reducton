from src.vector_store import VectorStore
from src.orchestrator import Orchestrator

vs = VectorStore()
vs.load()
orch = Orchestrator(vs)
query = "What are the legal rights of private defence (self-defence) under Bharatiya Nyaya Sanhita?"
res = orch.retriever.run(query, top_k=5)
print(f"Num results: {res['num_results']}")
for i, r in enumerate(res['results']):
    print(f"\n=== CHUNK {i+1} (score: {r['similarity_score']:.3f}, doc: {r['document_name']}) ===")
    print(r['text'][:500])
    print("...")
