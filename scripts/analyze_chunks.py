# scripts/analyze_chunks.py
import json;
from collections import Counter

with open("data/processed/chunks.json") as f:
    chunks = json.load(f)
sizes = [len(c["text"]) for c in chunks]
print(f"Total: {len(chunks)}, Min: {min(sizes)}, Max: {max(sizes)}, Avg: {sum(sizes)//len(sizes)}")

for label, lo, hi in [("<200",0,200),("200-500",200,500),("500-1k",500,1000),("1k-1.5k",1000,1500),(">1.5k",1500,999999)]:
    cnt = sum(1 for s in sizes if lo <= s < hi)
    print(f"  {label:>8s}: {cnt:4d} ({cnt*100//len(sizes):2d}%)")
print(f"Types: {dict(Counter(c['doc_type'] for c in chunks))}")

print("\n=== Oversized chunks ===")
for c in chunks:
    if len(c["text"]) > 1500:
        print(f"  {c['source']:30s} [{c['heading_path'][:50]}] {len(c['text'])} chars")
        print(f"    {c['text'][:100]}...")
        print()