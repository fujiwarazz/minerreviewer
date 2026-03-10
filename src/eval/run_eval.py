from __future__ import annotations

import random

from common.types import ArbiterOutput
from eval.metrics import average_rating, strength_weakness_balance
from pipeline.review_pipeline import ReviewPipeline
from storage.doc_store import DocStore


def run_evaluation(config_path: str, target_year: int, sample_size: int = 5) -> dict[str, float]:
    pipeline = ReviewPipeline(config_path)
    doc_store = DocStore()
    papers = doc_store.load_papers(pipeline.venue_id)
    candidates = [paper for paper in papers if paper.year == target_year]
    random.shuffle(candidates)
    sample = candidates[:sample_size]
    outputs: list[ArbiterOutput] = []
    for paper in sample:
        outputs.append(pipeline._run_review(paper, target_year))
    balance_scores = [strength_weakness_balance(output) for output in outputs]
    avg_balance = sum(balance_scores) / len(balance_scores) if balance_scores else 0.0
    return {
        "samples": len(outputs),
        "avg_rating": average_rating(outputs),
        "avg_strength_balance": avg_balance,
    }
