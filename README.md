# Production RAG: Documentation QA System

A production-grade Retrieval-Augmented Generation system that answers developer questions 
by retrieving relevant information from HuggingFace documentation. Features hybrid search 
(dense embeddings + BM25), cross-encoder re-ranking, and automated evaluation using RAGAS.

Built as an end-to-end ML engineering project covering data pipeline, retrieval optimization, 
LLM generation, systematic evaluation, and API deployment.

## Architecture

User Query
  → Query Processing
  → Hybrid Retrieval (BGE-base dense + BM25 sparse → RRF fusion)
  → Cross-encoder Re-ranking (top-20 → top-5)
  → LLM Generation (context assembly + answer)
  → Response with source citations

## Tech Stack

- **Embedding**: BAAI/bge-base-en-v1.5 (768d)
- **Vector Store**: ChromaDB
- **Sparse Search**: BM25 (rank_bm25)
- **Re-ranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Generator**: flan-t5-large (local) / Claude API (production)
- **Evaluation**: RAGAS, ROUGE-L, BERTScore
- **API**: FastAPI + Docker
- **Chunking**: heading-aware splitting, 1500 chars, 200 overlap

## Project Structure
```
production-rag/
├── config.py              # Centralized configuration
├── src/
│   ├── data_pipeline.py   # Ingestion, cleaning, chunking
│   ├── indexer.py         # Dense + sparse index building
│   ├── retriever.py       # Hybrid search + re-ranking
│   ├── generator.py       # Prompt assembly + LLM
│   ├── evaluator.py       # RAGAS + metrics
│   └── rag_pipeline.py    # End-to-end pipeline
├── api/app.py             # FastAPI endpoint
├── data/eval/             # Evaluation Q&A pairs
└── Dockerfile
```


## Results

| Metric | Baseline | Final | Delta |
|--------|----------|-------|-------|
| Faithfulness | TBD | TBD | TBD |
| Context Precision | TBD | TBD | TBD |
| Context Recall | TBD | TBD | TBD |
| Answer Relevance | TBD | TBD | TBD |
| MRR@5 | TBD | TBD | TBD |

## Key Design Decisions

1. **Hybrid retrieval over dense-only**: BM25 catches exact keyword matches that dense embeddings miss (e.g., function names, error codes)
2. **Two-stage retrieval**: Recall stage prioritizes coverage (top-50), re-ranking stage prioritizes precision (top-5) — balances latency vs accuracy
3. **Config-driven experimentation**: All hyperparameters in config.py for rapid A/B testing
4. **Evaluation-first development**: Eval dataset and baseline metrics defined before any optimization code
5. **Chunk size 1500 chars**: Fits within bge-base 512 token limit while preserving enough context per chunk