"""Indexer: builds
Implement on Day 31
"""

class Indexer:
    """Builds dense (vector) and sparse (BM25) indexes."""

    def __init__(self, config):
        self.config = config
        self.dense_index = None
        self.sparse_index = None
        self.chunks = []


    def build_dense_index(self, chunks: list) -> None:
        """Encode chunks wiht bge-base, store in ChromaDB/FAISS
        Day 31 implement
        """
        pass

    def build_sparse_index(self, chunks: list) -> None:
        """Build BM25 index from chunk texts.
        Day 31: implement with rank_bm25
        """
        pass

    def add_documents(self, new_chunks: list) -> None:
        """Incrementally add new chunks to existing indexes.
        Dense (ChromaDB): supports native add.
        Sparse (BM25): requires full rebuild (no incremental support).
        """
        pass

    def delete_by_source(self, source: str) -> None:
        """Remove all chunks from a specific source document.
        Used when a document is updated or removed.
        """
        pass

    def save(self, path: str) -> None:
        """Persist indexes to disk."""
        pass

    def load(self, path: str) -> None:
        pass
