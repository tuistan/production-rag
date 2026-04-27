# Architecture & Technical Decisions

## Module Overview

| Module | Responsibility | Implementation Day | Build vs Buy |
|--------|---------------|-------------------|--------------|
| Data Pipeline | Ingest, clean, chunk docs | Day 30 | Build (custom cleaning + heading-aware chunking) |
| Indexer | Embed chunks, build indexes | Day 31 | Buy (sentence-transformers, ChromaDB, rank_bm25) |
| Retriever | Hybrid search + re-rank | Day 31 | Build (RRF fusion logic) + Buy (cross-encoder) |
| Generator | Prompt assembly + LLM call | Day 32 | Build (prompt template) + Buy (flan-t5 / Claude API) |
| Evaluator | Automated eval pipeline | Day 32 | Build (eval loop) + Buy (RAGAS, rouge-score) |
| API | REST endpoint + streaming | Day 33 | Buy (FastAPI) |

## Data Pipeline Decisions

**Data Source: HuggingFace Documentation**
- Why: clean markdown/HTML, structured headings, code examples included
- Scope: ~100 pages covering Transformers, Datasets, Trainer, PEFT
- Alternative considered: SEC filings — rejected due to complex formatting

**Chunking Strategy: heading-aware, 1500 chars, 200 overlap**
- Why 1500 chars (~400 tokens): bge-base max = 512 tokens, leaves headroom
- Why heading-aware: preserves semantic boundaries vs blind character splitting
- Why 200 overlap: prevents key information from being cut at boundaries
- Experiment planned: compare 1000 / 1500 / 2000 on eval metrics

## Retrieval Decisions

**Hybrid Search (Dense + BM25)**
- Dense catches semantic similarity ("How to save a model" → "serialization")
- BM25 catches exact matches ("AutoModelForSequenceClassification", error codes)
- RRF fusion: simple, parameter-free, no training needed

**Two-Stage Design (Recall → Re-rank)**
- Stage 1 (recall): fast, high coverage, top-50 from each → RRF → top-20
- Stage 2 (re-rank): slow but precise, cross-encoder scores top-20 → top-5
- Why not single-stage: cross-encoder is O(n) per query, can't score entire corpus

## Generation Decisions

**Dual Provider**
- Development: flan-t5-large (free, local, fast iteration)
- Production: Claude/OpenAI API (better quality, handles complex questions)
- Config-driven switch: change one line in config.py

## Evaluation Decisions

**Evaluation-Driven Development**
- Eval dataset created before optimization code (Day 29)
- 40 Q&A pairs: factual / code / comparison / hallucination tests
- Baseline established Day 32, all improvements measured against it
- RAGAS 4 dimensions: Context Precision, Context Recall, Faithfulness, Answer Relevance