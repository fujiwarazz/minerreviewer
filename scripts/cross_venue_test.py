#!/usr/bin/env python3
"""Cross-venue evaluation script for memory-driven reviewer.

Tests generalization by training on one venue (ICLR) and testing on another (NeurIPS).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.logging import setup_logging
from common.types import Paper
from pipeline.review_pipeline import ReviewPipeline
from storage.parquet_loader import load_parquet_paper, load_parquet_ground_truth

logger = logging.getLogger(__name__)


# Mapping of venue names to parquet files
DATA_ROOT = Path("/mnt/data/zzh/datasets/crosseval/crosseval_std")


def find_parquet_file(venue: str, year: int) -> Path | None:
    """Find parquet file for venue and year."""
    file_path = DATA_ROOT / f"{venue}_{year}.parquet"
    if file_path.exists():
        return file_path
    return None


def infer_rating_from_decision(decision: str | None) -> float | None:
    """Infer rating from decision string."""
    if not decision:
        return None
    dec_lower = decision.lower()
    if 'accept' in dec_lower:
        if 'oral' in dec_lower or 'spotlight' in dec_lower:
            return 8.0
        elif 'poster' in dec_lower:
            return 7.0
        else:
            return 6.5
    elif 'reject' in dec_lower:
        return 4.0
    elif 'workshop' in dec_lower:
        return 5.0
    else:
        return None


def load_test_papers(parquet_path: Path, limit: int = 5, offset: int = 0) -> list[tuple[Paper, dict]]:
    """Load test papers from parquet file."""
    df = pd.read_parquet(parquet_path)
    papers = []
    for idx in range(offset, min(offset + limit, len(df))):
        row = df.iloc[idx]
        paper_id = str(row.get("paper_id") or row.get("id") or f"{parquet_path.stem}_{idx}")
        venue = str(row.get("venue") or row.get("venue_id") or "Unknown")
        year = int(row.get("year")) if pd.notna(row.get("year")) else None
        title = str(row.get("title") or "")
        abstract = str(row.get("abstract") or "")
        decision = str(row.get("decision") or "") if pd.notna(row.get("decision")) else None

        paper = Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            venue_id=venue,
            year=year,
            authors=[],
            fulltext="",
        )

        ground_truth = {
            "decision": decision,
            "inferred_rating": infer_rating_from_decision(decision),
        }
        papers.append((paper, ground_truth))
    return papers


def run_cross_venue_test(
    config_path: str,
    test_venue: str,
    test_year: int,
    memory_venue: str | None = None,  # 指定使用哪个 venue 的记忆
    limit: int = 5,
    offset: int = 0,
) -> dict:
    """Run cross-venue evaluation.

    Args:
        memory_venue: 如果指定，只从该 venue 的案例中检索；否则使用全部记忆
    """
    test_file = find_parquet_file(test_venue, test_year)
    if not test_file:
        raise FileNotFoundError(f"Test file not found for {test_venue} {test_year}")

    pipeline = ReviewPipeline(config_path)
    logger.info(f"Loaded pipeline with venue_id={pipeline.venue_id}")

    test_papers = load_test_papers(test_file, limit=limit, offset=offset)
    logger.info(f"Loaded {len(test_papers)} test papers from {test_file}")

    results = []
    decision_correct = 0
    rating_errors = []

    for i, (paper, gt) in enumerate(test_papers):
        logger.info(f"[{i+1}/{len(test_papers)}] {paper.title[:60]}...")
        logger.info(f"  GT Decision: {gt['decision']}")

        try:
            # Retrieve similar cases from specified venue's memory
            query_text = f"{paper.title}\n{paper.abstract}"
            case_results = pipeline.case_store.retrieve_cases(
                query_text=query_text,
                signature=None,
                top_k=5,
                venue_id=memory_venue,  # Use specified memory venue
                use_hybrid=True,
                exclude_paper_id=paper.paper_id,
                before_year=test_year,
            )
            similar_cases = [case for case, _ in case_results]
            logger.info(f"  Retrieved {len(similar_cases)} cases from {memory_venue or 'all venues'} memory")

            # Build retrieval bundle manually
            from common.types import RetrievalBundle
            bundle = RetrievalBundle(
                target_paper=paper,
                similar_paper_cases=similar_cases,
                policy_cards=[],  # Skip policy cards for simplicity
                venue_policy=None,
            )

            # Run review with custom bundle
            # Parse signature
            signature = pipeline._parse_paper(paper)

            # Mine criteria (use default)
            content_criteria, policy_criteria = pipeline._mine_criteria(paper, bundle, test_year)

            # Plan criteria
            activated = pipeline._plan_criteria(signature, bundle, content_criteria, policy_criteria)

            # Rewrite criteria (convert ActivatedCriterion to Criterion)
            criteria = pipeline._rewrite_criteria(paper, activated)

            # Run theme agents
            theme_outputs = pipeline._run_theme_agents(paper, criteria)

            # Aggregate with similar cases
            arbiter_output = pipeline._aggregate(
                theme_outputs, bundle.policy_cards, policy_criteria, bundle.venue_policy,
                similar_cases=similar_cases,
            )

            # Skip verification for faster testing

            # Extract predicted rating and decision
            pred_rating = arbiter_output.raw_rating
            pred_decision = arbiter_output.decision_recommendation

            # Compare with ground truth
            gt_decision_binary = "accept" if gt["decision"] and "accept" in gt["decision"].lower() else "reject"
            pred_decision_binary = "accept" if pred_decision and "accept" in pred_decision.lower() else "reject"
            decision_match = gt_decision_binary == pred_decision_binary

            if decision_match:
                decision_correct += 1

            if gt["inferred_rating"] and pred_rating:
                rating_errors.append(abs(pred_rating - gt["inferred_rating"]))

            logger.info(f"  Predicted: rating={pred_rating:.1f}, decision={pred_decision}")
            logger.info(f"  GT inferred_rating={gt['inferred_rating']}, decision_match={decision_match}")

            results.append({
                "paper_id": paper.paper_id,
                "title": paper.title[:60],
                "gt_decision": gt["decision"],
                "gt_inferred_rating": gt["inferred_rating"],
                "pred_rating": pred_rating,
                "pred_decision": pred_decision,
                "decision_match": decision_match,
                "strengths": arbiter_output.strengths[:200] if arbiter_output.strengths else None,
                "weaknesses": arbiter_output.weaknesses[:200] if arbiter_output.weaknesses else None,
            })

        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({
                "paper_id": paper.paper_id,
                "title": paper.title[:60],
                "error": str(e),
            })

    # Calculate metrics
    total = len(test_papers)
    decision_acc = decision_correct / total if total > 0 else 0
    mae = mean(rating_errors) if rating_errors else None
    rating_std = stdev(rating_errors) if len(rating_errors) > 1 else None

    summary = {
        "test_venue": test_venue,
        "test_year": test_year,
        "total_papers": total,
        "decision_accuracy": decision_acc,
        "decision_correct": decision_correct,
        "rating_mae": mae,
        "rating_std": rating_std,
        "results": results,
    }

    # Save results
    output_path = Path("data/processed/cross_venue_test_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n=== Summary ===")
    logger.info(f"Decision Accuracy: {decision_acc:.2%} ({decision_correct}/{total})")
    if mae:
        logger.info(f"Rating MAE: {mae:.2f} (std: {rating_std:.2f})" if rating_std else f"Rating MAE: {mae:.2f}")

    return summary


def main():
    setup_logging()
    import argparse

    parser = argparse.ArgumentParser(description="Cross-venue evaluation")
    parser.add_argument("--config", default="configs/iclr.yaml")
    parser.add_argument("--test_venue", default="NeurIPS")
    parser.add_argument("--test_year", type=int, default=2024)
    parser.add_argument("--memory_venue", default=None, help="Venue to use for memory retrieval (default: use all venues)")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)

    args = parser.parse_args()

    run_cross_venue_test(
        config_path=args.config,
        test_venue=args.test_venue,
        test_year=args.test_year,
        memory_venue=args.memory_venue,
        limit=args.limit,
        offset=args.offset,
    )


if __name__ == "__main__":
    main()