import sys; sys.path.insert(0, ".")
from config import config
from src.data_pipeline import DataPipeline
from collections import Counter

pipeline = DataPipeline(config)
docs = pipeline.ingest()
all_chunks = []
for doc in docs:
    cleaned = pipeline.clean(doc)
    chunks = pipeline.chunk(cleaned)
    all_chunks.extend(chunks)

print(f"Documents: {len(docs)}, Total chunks: {len(all_chunks)}")
sizes = [len(c.text) for c in all_chunks]
print(f"Sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
for src, cnt in Counter(c.source for c in all_chunks).most_common(10):
    print(f"  {src:40s} {cnt:4d} chunks")
for i, ch in enumerate(all_chunks[:3]):
    print(f"\n--- Chunk {i} ({ch.source}) ---")
    print(f"Heading: {ch.heading_path}")
    print(ch.text[:300])

print("\nSize distribution:")
for label, lo, hi in [("<200", 0, 200), ("200-500", 200, 500),
                       ("500-1k", 500, 1000), ("1k-1.5k", 1000, 1500),
                       (">1.5k", 1500, 999999)]:
    cnt = sum(1 for s in sizes if lo <= s < hi)
    pct = cnt * 100 // len(sizes)
    bar = "#" * (cnt * 40 // len(sizes))
    print(f"  {label:>8s}: {cnt:4d} ({pct:2d}%) {bar}")