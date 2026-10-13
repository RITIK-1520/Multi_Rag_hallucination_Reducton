from src.vector_store import VectorStore
import re

vs = VectorStore()
vs.load()

targets = ['Section 103', 'Section 303', 'Section 34', 'Section 111', 'Section 106']

for target in targets:
    print(f"\n==================== {target} ====================")
    matches = [c for c in vs.chunks if target.lower() in c['text'].lower() and c['document_name'] == 'Final_BNS']
    for idx, m in enumerate(matches[:3]):
        clean = m['text'].replace('\n', ' ')
        clean = re.sub(r'\s+', ' ', clean)
        print(f"[{idx+1}] {clean[:350]}...\n")
