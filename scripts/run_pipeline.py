# scripts/run_pipeline.py
import sys, logging; sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO)
from config import config
from src.data_pipeline import DataPipeline

pipeline = DataPipeline(config)
chunks = pipeline.run()
print(f"Total chunks: {len(chunks)}")
loaded = DataPipeline.load_chunks(f"{config.processed_data_dir}/chunks.json")
assert len(loaded) == len(chunks), "Mismatch!"
print("Save/load round-trip verified")