import sys
sys.path.insert(0, ".")
from config import config
from src.data_pipeline import DataPipeline

pipeline = DataPipeline(config)
docs = pipeline.ingest()

# 挑 trainer.md 测试（文件短，heading 结构清晰）
doc = [d for d in docs if d.source == "trainer.md"][0]
cleaned = pipeline.clean(doc)

print("=== Split by headings ===")
sections = pipeline._split_by_headings(cleaned.content)

print(f"\n=== Result: {len(sections)} sections ===")
for path, content in sections:
    preview = content[:80].replace("\n", " ")
    print(f"  [{path}] ({len(content)} chars) {preview}...")