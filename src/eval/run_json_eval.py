"""
Run evaluation comparing generated reviews against ground truth from JSON data.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from common.types import ArbiterOutput, Paper, Review
from common.utils import read_json, read_yaml
from eval.ground_truth_eval import evaluate_reviews, format_evaluation_report
from pipeline.review_pipeline import ReviewPipeline
from storage.doc_store import DocStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_papers_and_reviews(venue_id: str, doc_store: DocStore) -> tuple[list[Paper], list[Review]]:
    """Load papers and reviews from JSON files."""
    papers = doc_store.load_papers(venue_id)
    reviews = doc_store.load_reviews(venue_id)
    logger.info("Loaded %d papers and %d reviews", len(papers), len(reviews))
    return papers, reviews


def build_ground_truth_by_paper(paper_id: str, reviews: list[Review]) -> list[dict]:
    """Build ground truth list for a paper from its reviews."""
    paper_reviews = [r for r in reviews if r.paper_id == paper_id]
    gt_list = []
    for r in paper_reviews:
        gt = {
            "rating": r.rating,
            "decision": r.decision,
            "text": r.text,
        }
        # Extract strengths and weaknesses from text
        strengths, weaknesses = extract_sections_from_text(r.text)
        if strengths:
            gt["strengths"] = strengths
        if weaknesses:
            gt["weaknesses"] = weaknesses
        gt_list.append(gt)
    return gt_list


def extract_sections_from_text(text: str) -> tuple[str | None, str | None]:
    """Extract strengths and weaknesses from review text."""
    import re
    strengths = None
    weaknesses = None

    # Try various patterns
    strength_match = re.search(
        r'\*\*strengths?\*\*:?\s*(.*?)(?=\*\*weaknesses?\*\*|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    if strength_match:
        strengths = strength_match.group(1).strip()

    weakness_match = re.search(
        r'\*\*weaknesses?\*\*:?\s*(.*?)(?=\*\*|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    if weakness_match:
        weaknesses = weakness_match.group(1).strip()

    # Alternative patterns
    if not strengths:
        strength_match = re.search(
            r'(?:pros|strengths|positive)[s:]?\s*(.*?)(?=(?:cons|weaknesses|negative|$))',
            text, re.IGNORECASE | re.DOTALL
        )
        if strength_match:
            strengths = strength_match.group(1).strip()

    if not weaknesses:
        weakness_match = re.search(
            r'(?:cons|weaknesses|negative)[s:]?\s*(.*?)(?=$)',
            text, re.IGNORECASE | re.DOTALL
        )
        if weakness_match:
            weaknesses = weakness_match.group(1).strip()

    return strengths, weaknesses


def run_evaluation(
    config_path: str,
    sample_size: int = 50,
    output_path: str | None = None,
    random_seed: int = 42,
) -> None:
    """Run evaluation on sampled papers."""

    # Initialize
    pipeline = ReviewPipeline(config_path)
    config = read_yaml(config_path)
    embedding_client = EmbeddingClient(EmbeddingConfig(**config["embedding"]))
    doc_store = DocStore()

    # Load data
    papers, reviews = load_papers_and_reviews(pipeline.venue_id, doc_store)

    # Filter papers that have reviews with rating
    papers_with_reviews = {}
    for p in papers:
        paper_reviews = [r for r in reviews if r.paper_id == p.paper_id and r.rating is not None]
        if paper_reviews:
            papers_with_reviews[p.paper_id] = p

    logger.info("Papers with rated reviews: %d", len(papers_with_reviews))

    # Sample papers
    random.seed(random_seed)
    sampled_ids = random.sample(list(papers_with_reviews.keys()), min(sample_size, len(papers_with_reviews)))
    sampled_papers = [papers_with_reviews[pid] for pid in sampled_ids]

    print(f"\n{'='*60}")
    print(f"Evaluating {len(sampled_papers)} papers...")
    print(f"{'='*60}\n")

    # Process papers
    outputs: list[ArbiterOutput] = []
    ground_truths: list[list[dict]] = []
    processing_times: list[float] = []

    for i, paper in enumerate(sampled_papers):
        try:
            print(f"[{i+1}/{len(sampled_papers)}] Processing: {paper.title[:60]}...")

            start_time = time.time()
            output = pipeline._run_review(paper, paper.year)
            elapsed = time.time() - start_time

            outputs.append(output)
            processing_times.append(elapsed)

            # Build ground truth from reviews
            gt_list = build_ground_truth_by_paper(paper.paper_id, reviews)
            ground_truths.append(gt_list if gt_list else [{}])

            print(f"  Rating: {output.raw_rating}, Decision: {output.decision_recommendation}, Time: {elapsed:.1f}s")

        except Exception as e:
            logger.error("Failed to process paper %s: %s", paper.paper_id, e)
            import traceback
            traceback.print_exc()
            continue

    if not outputs:
        print("ERROR: No papers were successfully evaluated")
        return

    # Evaluate
    print(f"\n{'='*60}")
    print("Computing evaluation metrics...")
    print(f"{'='*60}\n")

    result = evaluate_reviews(
        outputs,
        ground_truths,
        embedding_client,
        similarity_threshold=0.5,
    )

    # Print report
    print(format_evaluation_report(result))

    # Print timing statistics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"\n## Processing Time")
        print(f"  Total: {sum(processing_times):.1f}s")
        print(f"  Average per paper: {avg_time:.1f}s")
        print(f"  Min: {min(processing_times):.1f}s")
        print(f"  Max: {max(processing_times):.1f}s")

    # Save results
    if output_path:
        result_dict = {
            "rating_mae": result.rating_mae,
            "rating_mse": result.rating_mse,
            "rating_correlation": result.rating_correlation,
            "decision_accuracy": result.decision_accuracy,
            "decision_f1_accept": result.decision_f1_accept,
            "decision_f1_reject": result.decision_f1_reject,
            "strength_coverage": result.strength_coverage,
            "weakness_coverage": result.weakness_coverage,
            "text_similarity": result.text_similarity,
            "samples_evaluated": result.samples_evaluated,
            "samples_with_gt_rating": sum(1 for d in result.details if d.get("gt_rating") is not None),
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "details": result.details,
        }
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated reviews against ground truth")
    parser.add_argument("--config", default="configs/iclr.yaml", help="Config file path")
    parser.add_argument("--sample_size", type=int, default=50, help="Number of papers to evaluate")
    parser.add_argument("--output", default="data/processed/eval_results.json", help="Output path for results JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    run_evaluation(
        args.config,
        args.sample_size,
        args.output,
        args.seed,
    )


if __name__ == "__main__":
    main()