import sys; sys.path.insert(0, ".")
import logging
logging.basicConfig(level=logging.INFO)

from config import config
from src.data_pipeline import DataPipeline
from src.indexer import Indexer

chunks = DataPipeline.load_chunks(f"{config.processed_data_dir}/chunks.json")
print(f"Loaded {len(chunks)} chunks")

indexer = Indexer(config)
indexer.build(chunks)

# Smoke test
print("Dense count:", indexer.collection.count())
print("BM25 corpus size:", len(indexer.bm25.doc_freqs))