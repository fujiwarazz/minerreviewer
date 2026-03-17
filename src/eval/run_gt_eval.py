"""
Run evaluation comparing generated reviews against ground truth.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from common.utils import read_yaml
from eval.ground_truth_eval import evaluate_reviews, format_evaluation_report
from pipeline.review_pipeline import ReviewPipeline
from storage.parquet_loader import load_parquet_ground_truth, load_parquet_paper

logger = logging.getLogger(__name__)


def run_ground_truth_evaluation(
    config_path: str,
    parquet_path: str,
    sample_size: int = 10,
    output_path: str | None = None,
) -> None:
    """
    Run evaluation comparing generated reviews to ground truth.

    Args:
        config_path: Path to config YAML
        parquet_path: Path to parquet file with ground truth
        sample_size: Number of papers to evaluate
        output_path: Optional path to save results JSON
    """
    pipeline = ReviewPipeline(config_path)
    config = read_yaml(config_path)

    embedding_client = EmbeddingClient(EmbeddingConfig(**config["embedding"]))

    outputs = []
    ground_truths = []

    # Load papers and run reviews
    for row_idx in range(sample_size):
        try:
            paper = load_parquet_paper(parquet_path, row_idx, venue_id=pipeline.venue_id)
            if paper is None:
                logger.warning("Could not load paper at row %d", row_idx)
                continue

            logger.info("Reviewing paper %d: %s", row_idx, paper.title[:50])
            print(f"Processing paper {row_idx + 1}/{sample_size}: {paper.title[:50]}...")

            # Generate review
            output = pipeline._run_review(paper, paper.year)
            outputs.append(output)

            # Load ground truth (returns list of reviews for this paper)
            # Pass the entire list so evaluation can merge rating + content
            gt_list = load_parquet_ground_truth(parquet_path, row_idx)
            ground_truths.append(gt_list if gt_list else [{}])

        except Exception as e:
            logger.error("Failed to process row %d: %s", row_idx, e)
            import traceback
            traceback.print_exc()
            continue

    if not outputs:
        logger.error("No papers were successfully evaluated")
        print("ERROR: No papers were successfully evaluated")
        return

    # Evaluate
    result = evaluate_reviews(
        outputs,
        ground_truths,
        embedding_client,
        similarity_threshold=0.5,
    )

    # Print report
    print(format_evaluation_report(result))

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
            "details": result.details,
        }
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        logger.info("Results saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated reviews against ground truth")
    parser.add_argument("--config", default="configs/iclr.yaml", help="Config file path")
    parser.add_argument("--parquet", required=True, help="Parquet file with ground truth")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of papers to evaluate")
    parser.add_argument("--output", help="Output path for results JSON")
    args = parser.parse_args()

    run_ground_truth_evaluation(
        args.config,
        args.parquet,
        args.sample_size,
        args.output,
    )


if __name__ == "__main__":
    main()