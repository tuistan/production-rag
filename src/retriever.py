"""Retrieval Module: hybrid search (dense + BM25 via RRF) -> cross encoder re-rank."""
import logging
from typing import Optional

from sentence_transformers import CrossEncoder
from sentence_transformers.util import normalize_embeddings
from sknetwork.ranking import top_k

logger = logging.getLogger(__name__)


class Retriever:
    """Two-stage retrieval:
    Stage 1 hybrid recall from dense + BM25,
    Stage 2 cross-encoder rerank.
    """

    def __init__(self, config, indexer):
        self.config = config
        self.indexer = indexer
        self.reranker: Optional[CrossEncoder] = None


    # ---------- public ----------

    def retrieve(self, query: str) -> list[dict]:
        """Full retrieval: hybrid_search -> rerank."""
        candidates = self.hybrid_search(query)
        return self.rerank(query, candidates)

    # ---------- Stage 1a: dense ----------

    def dense_search(self, query: str, top_k: int = None) -> list[dict]:
        """Stage 1a: dense embedding similarity search"""
        top_k = top_k or self.config.top_k_recall
        self.indexer._ensure_embedder()
        q_emb = self.indexer.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()
        result = self.indexer.collection.query(
            query_embeddings=[q_emb], n_results=top_k,
        )
        out = []
        for i in range(len(result["ids"][0])):
            out.append({
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "score": 1.0 - result["distances"][0][i],
                "rank": i + 1,
            })
        return out

    # ---------- Stage 1b: BM25/sparse ----------

    def sparse_search(self, query: str, top_k: int = None) -> list[dict]:
        """Stage 1b: BM25 keyword search"""
        top_k  = top_k or self.config.top_k_recall
        tokens = query.lower().split()
        scores = self.indexer.bm25.get_scores(tokens) # np.array, len = corpus_size
        # top-k indices
        top_indices = scores.argsort()[::-1][:top_k]
        out = []
        for rank, idx in enumerate(top_indices, start=1):
            c = self.indexer.chunks[idx]
            out.append({
                "id": f"{c.source}::{c.chunk_index}",
                "text": c.text,
                "metadata": {
                    "source": c.source, "title": c.title,
                    "doc_type": c.doc_type, "heading_path": c.heading_path,
                    "chunk_index": c.chunk_index,
                },
                "score": float(scores[idx]),
                "rank": rank,
            })
        return out

    # ---------- Stage 1: hybrid via weighted RRF ----------

    def hybrid_search(self, query: str) -> list[dict]:
        """Weighted RRF over dense + sparse rankings.
        """
        dense = self.dense_search(query, top_k=self.config.top_k_recall)
        sparse = self.sparse_search(query, top_k=self.config.top_k_recall)

        k = self.config.rrf_k
        wd = self.config.dense_weight
        ws = self.config.bm25_weight

        agg = {}  # id -> {"text", "metadata", "score"}
        for r in dense:
            agg[r["id"]] = {
                "text": r["text"], "metadata": r["metadata"],
                "score": wd / (k +r["rank"]), "_dense_rank": r["rank"],
            }
        for r in sparse:
            if r["id"] in agg:
                agg[r["id"]]["score"] += ws / (k + r["rank"])
                agg[r["id"]]["_sparse_rank"] = r["rank"]
            else:
                agg[r["id"]] = {
                    "text": r["text"], "metadata": r["metadata"],
                    "score": ws / (k + r["rank"]), "_sparse_rank": r["rank"],
                }

        fused = sorted([{"id": id_, **v} for id_, v in agg.items()],
                       key = lambda x: x["score"], reverse=True)
        return fused[:self.config.top_k_fusion]

    # ---------- Stage 2: cross-encoder rerank ----------

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        if self.reranker is None:
            logger.info(f"Loading reranker: {self.config.reranker_model}")
            self.reranker = CrossEncoder(self.config.reranker_model)

        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self.reranker.predict(pairs, batch_size=32)

        # ce_scores is a raw logit, can be negative; only relative order matters
        for c, s in zip(candidates, ce_scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key = lambda x: x["rerank_score"], reverse=True)
        return ranked[:self.config.top_k_rerank]


