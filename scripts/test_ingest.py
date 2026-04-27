import sys; sys.path.insert(0, ".")
from config import config
from src.data_pipeline import DataPipeline

pipeline = DataPipeline(config)
docs = pipeline.ingest()

print(f"\nTotal docs: {len(docs)}")
for doc in docs[:5]:
    print(f"    {doc.source:40s} type={doc.doc_type:15s} "
          f"len={len(doc.content):,} chars")

doc = docs[0]
cleaned = pipeline.clean(doc)

print(f"\n=== {doc.source} ===")
print(f"Before: {len(doc.content):,} chars")
print(f"After:  {len(cleaned.content):,} chars")

# 看 clean 前的前 500 字符
print("\n--- BEFORE ---")
print(doc.content[:1000])

# 看 clean 后的前 500 字符
print("\n--- AFTER ---")
print(cleaned.content[:1000])