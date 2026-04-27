"""Evaluation Module: RAGAS metrics + automated testing.
Implemented on Day 32.
"""
import json


class Evaluator:
    """Evaluates RAG pipeline on eval dataset."""

    def __init__(self, config):
        self.config = config

    def load_eval_dataset(self, path: str = None) -> list[dict]:
        """Load Q&A pairs from eval_dataset.json."""
        if path is None:
            path = f"{self.config.eval_data_dir}/eval_dataset.json"
        with open(path) as f:
            return json.load(f)

    def evaluate_retrieval(self, query: str, retrieval_chunks: list[dict],
                           ground_truth_sources: list[str]) -> dict:
        """Retrieval metrics: MRR@5, Context Precision, Context Recall"""
        pass

    def evaluate_generation(self, prediction: str,
                            ground_truth: str, contexts: list[str]) -> dict:
        """Generation metrics: ROUGE-L, BERTScore, Faithfulness, Relevance."""
        pass

    def evaluate_pipeline(self, pipeline, eval_dataset: list[dict]) -> dict:
        """Return full evaluation on entire eval dataset.
        Returns aggregated metrics dict.
        """
        pass

    def compare_with_baseline(
            self, current_metrics: dict, baseline_metrics: dict) -> dict:
        """Compare current vs baseline, return the delta for each metric."""
        pass

