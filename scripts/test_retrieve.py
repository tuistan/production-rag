import sys; sys.path.insert(0, ".")
import logging
logging.basicConfig(level=logging.INFO)

from config import config
from src.data_pipeline import DataPipeline
from src.indexer import Indexer
from src.retriever import Retriever

chunks = DataPipeline.load_chunks(f"{config.processed_data_dir}/chunks.json")

indexer = Indexer(config)
indexer.build(chunks)

retriever = Retriever(config, indexer)

queries = [
    "How do I fine-tune a model with LoRA?",
    "What is the Trainer API?",
    "How to use AutoModelForSequenceClassification?",
    "What is the capital of France?",  # OOD test
]

for q in queries:
    print(f"\n=== Query: {q} ===")
    results = retriever.retrieve(q)
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"  [{i}] rerank={r['rerank_score']:+.2f}  "
              f"src={meta['source']}  path={meta['heading_path'][:50]}")
        print(f"      {r['text'][:100].strip()}...")