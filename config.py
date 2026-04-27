from dataclasses import dataclass
from pathlib import Path

# 项目根目录：config.py 所在的目录
PROJECT_ROOT = Path(__file__).parent


@dataclass
class RAGConfig:
    """
    Production RAG system configuration.
    All hyperparameters centralized here for rapid experimentation.
    """

    # --- Data Pipeline ---
    chunk_size: int = 1500  # ~400 tokens, bge-base max=512
    chunk_overlap: int = 200  # ~50 tokens overlap
    min_chunk_size: int = 100  # drop short chunks (header/footer remnants)

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # --- Retrieval ---
    top_k_recall: int = 50  # Stage 1: candidates before reranking
    top_k_rerank: int = 5  # Stage 2: final results after re-ranking
    bm25_weight: float = 0.3  # BM25 weight in RRF fusion
    dense_weight: float = 0.7  # Dense embedding weight in RRF fusion
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Generation ---
    llm_provider: str = "flan-t5"  # "flan-t5" | "openai" | "anthropic"
    llm_model: str = "google/flan-t5-large"
    max_context_length: int = 2048
    temperature: float = 0.1

    # --- Paths (absolute, based on project root) ---
    raw_data_dir: str = str(PROJECT_ROOT / "data" / "raw")
    processed_data_dir: str = str(PROJECT_ROOT / "data" / "processed")
    eval_data_dir: str = str(PROJECT_ROOT / "data" / "eval")
    index_dir: str = str(PROJECT_ROOT / "data" / "index")


config = RAGConfig()