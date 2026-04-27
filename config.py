from dataclasses import dataclass

@dataclass
class RAGConfig:
    """
    Production RAG system configuration.
    All hyperparameters centralized here for rapid experimentation.
    """

    # --- Data Pipeline ---
    chunk_size: int = 1500          # ~400 tokens, bgs-base max=512
    chunk_overlap: int = 200        # ~50 tokens overlap
    min_chunk_size: int = 100       # drop short chunks (header/footer remnants)

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # --- Retrieval ---
    top_k_recall: int = 50          # Stage 1:candidates to retrieve before reranking
    top_k_rerank: int = 5           # Stage 2: final results after re-ranking
    bm25_weight: float = 0.3        # BM25 weight in RFF fusion
    dense_weight: float = 0.7       # Dense embedding weight in RFF fusion
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


    # --- Generation ---
    llm_provider: str = "flan-t5"   # "flan-t5" | "openai" | "anthropic"
    llm_model: str = "google/flan-t5-large"
    max_context_length: int = 2048
    temperature: float = 0.1

    # --- Path ---
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    evel_data_dir: str = "data/evel"
    index_dir: str = "data/index"

config = RAGConfig()
