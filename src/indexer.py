"""Indexer: builds dense(ChromaDB) and sparse(BM25) from chunks."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Indexer:
    """Builds dense (vector) and sparse (BM25) indexes."""

    COLLECTION_NAME = "rag_chunks"

    def __init__(self, config):
        self.config = config
        self.embedder: Optional[SentenceTransformer] = None
        self.chroma_client = None
        self.collection = None
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: list = []

    # ---------- public API used by RAGPipeline ----------

    def build(self, chunks: list) -> None:
        """Full rebuild. Idempotent (drops existing collection)."""
        logger.info(f"Building indexes for {len(chunks)} chunks")
        self.chunks = list(chunks)
        self._build_dense(self.chunks)
        self._build_sparse(self.chunks)
        logger.info("Indexes built.")

    def add_chunks(self, new_chunks: list) -> None:
        """Incremental add. Dense supports native add, BM25 must rebuild."""
        if not new_chunks:
            return
        self._add_to_dense(new_chunks)
        self.chunks.extend(new_chunks)
        # BM25 has no incremental — full rebuild on the union
        self._build_sparse(self.chunks)
        logger.info(f"Added {len(new_chunks)} chunks; total={self.chunks}")

    def delete_by_source(self, source: str) -> None:
        """Remove all chunks whose source matches. Both indexes."""
        # Dense: ChromaDB metadata delete
        self.collection.delete(where={"source": source})
        # Sparse: filter and rebuild
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.source != source]
        self._build_sparse(self.chunks)
        logger.info(f"Deleted {before - len(self.chunks)} chunks from source={source}")

    # ---------- internals ----------
    def _ensure_embedder(self):
        if self.embedder is None:
            logger.info(f"Loading embedder: {self.config.embedding_model}")
            self.embedder = SentenceTransformer(self.config.embedding_model)

    def _ensure_chroma(self):
        if self.chroma_client is None:
            persist_dir = Path(self.config.index_dir) / "chroma"
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )

    def _build_dense(self, chunks: list) -> None:
        """Encode chunks with bge-base, store in ChromaDB/FAISS
        Day 31 implement
        """
        self._ensure_embedder()
        self._ensure_chroma()

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        existing_ids = self.collection.get()["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(
            texts,
            batch_sie=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

        ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "title": c.title,
                "doc_type": c.doc_type,
                "heading_path": c.heading_path,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        # Chroma can choke on >5461 items in one add, thus batch it
        BATCH = 1000
        for i in range(0, len(chunks), BATCH):
            self.collection.add(
                ids=ids[i : i + BATCH],
                embeddings=embeddings[i : i + BATCH],
                documents=texts[i : i + BATCH],
                metadatas=metadatas[i : i + BATCH],
            )
        logger.info(f"Dense index: {len(chunks)} chunks in ChromaDB")

    def _add_to_dense(self, new_chunks: list) -> None:
        self._ensure_embedder()
        self._ensure_chroma()

        if self.collection is None:
            self.collection = self.chroma_client.get_collection(self.COLLECTION_NAME)

        texts = [c.text for c in new_chunks]
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()
        ids = [f"{c.source}::{c.chunk_index}" for c in new_chunks]
        metadatas = [
            {
                "source": c.source,
                "title": c.title,
                "doc_type": c.doc_type,
                "heading_path": c.heading_path,
                "chunk_index": c.chunk_index,
            }
            for c in new_chunks
        ]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def _build_sparse(self, chunks: list) -> None:
        # BM25Okapi expects pre-tokenized: list[list[str]]
        # Cheap tokenization: lowercase + whitespace split. Good enough for English docs.
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"Sparse index: {len(chunks)} chunks in BM25")

    def save(self) -> None:
        """ChromaDB persists itself. Save BM25 + chunks list separately."""
        out_dir = Path(self.config.index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        logger.info(f"Saved BM25 + chunks list to {out_dir}")

    def load(self, path: str) -> None:
        out_dir = Path(self.config.index_dir)

        # ChromaDB
        self._ensure_chroma()
        self.collection = self.chroma_client.get_collection(self.COLLECTION_NAME)

        # BM25 + chunks
        with open(out_dir / "bm25.pkl") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        logger.info(f"Loaded indexes; {len(self.chunks)} chunks")
