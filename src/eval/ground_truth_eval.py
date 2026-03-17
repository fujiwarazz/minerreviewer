"""
Ground truth evaluation metrics for peer review system.

Compares generated reviews against real human reviews from OpenReview.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np

from clients.embedding_client import EmbeddingClient
from common.types import ArbiterOutput

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics."""
    # Rating metrics
    rating_mae: float  # Mean Absolute Error vs ground truth
    rating_mse: float  # Mean Squared Error
    rating_correlation: float | None  # Pearson correlation

    # Decision metrics
    decision_accuracy: float  # Accuracy of accept/reject prediction
    decision_f1_accept: float  # F1 for accept class
    decision_f1_reject: float  # F1 for reject class

    # Content metrics
    strength_coverage: float  # Coverage of ground truth strengths
    weakness_coverage: float  # Coverage of ground truth weaknesses
    text_similarity: float  # Overall semantic similarity

    # Summary
    samples_evaluated: int
    details: list[dict[str, Any]]


def evaluate_reviews(
    generated_outputs: list[ArbiterOutput],
    ground_truth_reviews: list[list[dict[str, Any]]],
    embedding_client: EmbeddingClient,
    similarity_threshold: float = 0.5,
) -> EvaluationResult:
    """
    Evaluate generated reviews against ground truth.

    Args:
        generated_outputs: List of generated ArbiterOutput
        ground_truth_reviews: List of lists of ground truth review dicts (same order as outputs).
                              Each paper may have multiple GT reviews with different fields.
        embedding_client: For computing semantic similarity
        similarity_threshold: Threshold for coverage matching

    Returns:
        EvaluationResult with all metrics
    """
    if len(generated_outputs) != len(ground_truth_reviews):
        raise ValueError(
            f"Mismatch: {len(generated_outputs)} outputs vs {len(ground_truth_reviews)} ground truth"
        )

    details = []
    rating_errors = []
    rating_squared_errors = []
    gt_ratings = []
    pred_ratings = []
    decision_correct = 0
    accept_tp = accept_fp = accept_fn = 0
    reject_tp = reject_fp = reject_fn = 0
    strength_coverages = []
    weakness_coverages = []
    text_similarities = []

    for i, (output, gt_list) in enumerate(zip(generated_outputs, ground_truth_reviews)):
        detail = {"index": i}

        # Merge all GT reviews for this paper - use rating from any GT that has it,
        # and strengths/weaknesses from any GT that has them
        merged_gt = _merge_ground_truths(gt_list)

        # Rating evaluation - use GT that has rating
        gt_rating = merged_gt.get("rating")
        pred_rating = output.raw_rating
        if gt_rating is not None and pred_rating is not None:
            error = abs(pred_rating - gt_rating)
            rating_errors.append(error)
            rating_squared_errors.append(error ** 2)
            gt_ratings.append(gt_rating)
            pred_ratings.append(pred_rating)
            detail["gt_rating"] = gt_rating
            detail["pred_rating"] = pred_rating
            detail["rating_error"] = error

        # Decision evaluation
        gt_decision = merged_gt.get("decision")
        pred_decision = _normalize_decision(output.decision_recommendation)
        if gt_decision and pred_decision:
            detail["gt_decision"] = gt_decision
            detail["pred_decision"] = pred_decision
            if gt_decision == pred_decision:
                decision_correct += 1
                detail["decision_correct"] = True
            else:
                detail["decision_correct"] = False

            # F1 calculation
            if gt_decision == "accept":
                if pred_decision == "accept":
                    accept_tp += 1
                else:
                    accept_fn += 1
            else:
                if pred_decision == "accept":
                    accept_fp += 1
                else:
                    reject_tp += 1

        # Content coverage evaluation - use GT that has strengths/weaknesses
        gt_strengths = merged_gt.get("strengths", [])
        gt_weaknesses = merged_gt.get("weaknesses", [])

        if gt_strengths:
            strength_cov = _compute_coverage(
                output.strengths, gt_strengths, embedding_client, similarity_threshold
            )
            strength_coverages.append(strength_cov)
            detail["strength_coverage"] = strength_cov

        if gt_weaknesses:
            weakness_cov = _compute_coverage(
                output.weaknesses, gt_weaknesses, embedding_client, similarity_threshold
            )
            weakness_coverages.append(weakness_cov)
            detail["weakness_coverage"] = weakness_cov

        # Text similarity
        all_gt = gt_strengths + gt_weaknesses
        all_pred = output.strengths + output.weaknesses
        if all_gt and all_pred:
            text_sim = _compute_text_similarity(all_gt, all_pred, embedding_client)
            text_similarities.append(text_sim)
            detail["text_similarity"] = text_sim

        details.append(detail)

    # Compute aggregate metrics
    n = len(generated_outputs)
    rating_mae = mean(rating_errors) if rating_errors else 0.0
    rating_mse = mean(rating_squared_errors) if rating_squared_errors else 0.0
    rating_correlation = (
        _pearson_correlation(gt_ratings, pred_ratings)
        if len(gt_ratings) >= 2
        else None
    )
    decision_accuracy = decision_correct / n if n > 0 else 0.0

    # F1 scores
    accept_precision = accept_tp / (accept_tp + accept_fp) if (accept_tp + accept_fp) > 0 else 0.0
    accept_recall = accept_tp / (accept_tp + accept_fn) if (accept_tp + accept_fn) > 0 else 0.0
    decision_f1_accept = (
        2 * accept_precision * accept_recall / (accept_precision + accept_recall)
        if (accept_precision + accept_recall) > 0
        else 0.0
    )

    reject_precision = reject_tp / (reject_tp + reject_fp) if (reject_tp + reject_fp) > 0 else 0.0
    reject_recall = reject_tp / (reject_tp + reject_fn) if (reject_tp + reject_fn) > 0 else 0.0
    decision_f1_reject = (
        2 * reject_precision * reject_recall / (reject_precision + reject_recall)
        if (reject_precision + reject_recall) > 0
        else 0.0
    )

    return EvaluationResult(
        rating_mae=round(rating_mae, 3),
        rating_mse=round(rating_mse, 3),
        rating_correlation=round(rating_correlation, 3) if rating_correlation else None,
        decision_accuracy=round(decision_accuracy, 3),
        decision_f1_accept=round(decision_f1_accept, 3),
        decision_f1_reject=round(decision_f1_reject, 3),
        strength_coverage=round(mean(strength_coverages), 3) if strength_coverages else 0.0,
        weakness_coverage=round(mean(weakness_coverages), 3) if weakness_coverages else 0.0,
        text_similarity=round(mean(text_similarities), 3) if text_similarities else 0.0,
        samples_evaluated=n,
        details=details,
    )


def _merge_ground_truths(gt_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge multiple ground truth reviews for a single paper.

    Some GT reviews have rating but no strengths/weaknesses,
    while others have strengths/weaknesses but no rating.
    This function merges them to get the best of both.
    """
    merged = {
        "rating": None,
        "decision": None,
        "strengths": [],
        "weaknesses": [],
    }

    for gt in gt_list:
        # Get rating from first GT that has it
        if merged["rating"] is None:
            merged["rating"] = _extract_gt_rating(gt)

        # Get decision from first GT that has it
        if merged["decision"] is None:
            merged["decision"] = _extract_gt_decision(gt)

        # Collect all strengths and weaknesses
        strengths = _extract_gt_points(gt, "strengths")
        weaknesses = _extract_gt_points(gt, "weaknesses")

        # Add unique points only (avoid duplicates)
        for s in strengths:
            if s not in merged["strengths"]:
                merged["strengths"].append(s)
        for w in weaknesses:
            if w not in merged["weaknesses"]:
                merged["weaknesses"].append(w)

    return merged


def _extract_gt_rating(gt: dict[str, Any]) -> float | None:
    """Extract rating from ground truth review."""
    rating = gt.get("rating")
    if rating is None:
        rating = gt.get("recommendation")
    if rating is None:
        return None
    if isinstance(rating, dict):
        rating = rating.get("value")
    if isinstance(rating, str):
        # Parse "7: Good paper, accept" format
        rating = rating.split(":")[0].strip()
    try:
        return float(rating)
    except (ValueError, TypeError):
        return None


def _extract_gt_decision(gt: dict[str, Any]) -> str | None:
    """Extract decision from ground truth review."""
    decision = gt.get("decision") or gt.get("recommendation")
    if decision is None:
        return None
    decision = str(decision).lower()
    if "accept" in decision:
        return "accept"
    if "reject" in decision:
        return "reject"
    return "borderline"


def _normalize_decision(decision: str | None) -> str | None:
    """Normalize decision string."""
    if decision is None:
        return None
    decision = str(decision).lower()
    if "accept" in decision:
        return "accept"
    if "reject" in decision:
        return "reject"
    if "revise" in decision or "borderline" in decision:
        return "borderline"
    return None


def _extract_gt_points(gt: dict[str, Any], field: str) -> list[str]:
    """Extract strengths/weaknesses from ground truth."""
    value = gt.get(field)
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.get("value", "")
    if not value:
        return []

    # Split by common delimiters
    text = str(value)
    points = []
    for part in text.split("@@"):
        part = part.strip()
        if part:
            points.append(part)
    if len(points) <= 1:
        # Try splitting by newlines
        points = [p.strip(" -*\t") for p in text.splitlines() if p.strip(" -*\t")]
    return points[:15]  # Limit points


def _compute_coverage(
    pred_points: list[str],
    gt_points: list[str],
    embedding_client: EmbeddingClient,
    threshold: float,
) -> float:
    """Compute coverage of ground truth points by predictions."""
    if not gt_points or not pred_points:
        return 0.0

    try:
        gt_embs = embedding_client.embed(gt_points)
        pred_embs = embedding_client.embed(pred_points)

        # Normalize embeddings
        gt_norm = gt_embs / (np.linalg.norm(gt_embs, axis=1, keepdims=True) + 1e-9)
        pred_norm = pred_embs / (np.linalg.norm(pred_embs, axis=1, keepdims=True) + 1e-9)

        # Compute cosine similarity
        sim = np.dot(gt_norm, pred_norm.T)
        best_sim = np.max(sim, axis=1)

        # Count matches above threshold
        matches = np.sum(best_sim >= threshold)
        return float(matches / len(gt_points))
    except Exception as e:
        logger.warning("Coverage computation failed: %s", e)
        return 0.0


def _compute_text_similarity(
    texts1: list[str],
    texts2: list[str],
    embedding_client: EmbeddingClient,
) -> float:
    """Compute overall semantic similarity between two sets of texts."""
    if not texts1 or not texts2:
        return 0.0

    try:
        # Compute mean embeddings
        emb1 = embedding_client.embed(texts1)
        emb2 = embedding_client.embed(texts2)

        mean1 = np.mean(emb1, axis=0)
        mean2 = np.mean(emb2, axis=0)

        # Cosine similarity
        sim = np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2) + 1e-9)
        return float(sim)
    except Exception as e:
        logger.warning("Similarity computation failed: %s", e)
        return 0.0


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    x_arr = np.array(x)
    y_arr = np.array(y)

    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)

    numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
    denominator = np.sqrt(np.sum((x_arr - x_mean) ** 2) * np.sum((y_arr - y_mean) ** 2))

    if denominator < 1e-9:
        return 0.0
    return float(numerator / denominator)


def format_evaluation_report(result: EvaluationResult) -> str:
    """Format evaluation result as a readable report."""

    # Check if rating metrics are valid (non-zero samples)
    rating_valid = result.samples_evaluated > 0 and any(
        d.get("gt_rating") is not None for d in result.details
    )

    lines = [
        "=" * 60,
        "EVALUATION REPORT",
        "=" * 60,
        "",
        "## Rating Metrics",
        f"  MAE (Mean Absolute Error): {result.rating_mae if rating_valid else 'N/A'}",
        f"  MSE (Mean Squared Error):  {result.rating_mse if rating_valid else 'N/A'}",
        f"  Correlation:               {result.rating_correlation if result.rating_correlation else 'N/A'}",
        "",
        "## Decision Metrics",
        f"  Accuracy:                  {result.decision_accuracy}",
        f"  F1 (Accept):               {result.decision_f1_accept}",
        f"  F1 (Reject):               {result.decision_f1_reject}",
        "",
        "## Content Metrics",
        f"  Strength Coverage:         {result.strength_coverage}",
        f"  Weakness Coverage:         {result.weakness_coverage}",
        f"  Text Similarity:           {result.text_similarity}",
        "",
        f"  Samples Evaluated:         {result.samples_evaluated}",
        f"  Samples with GT Rating:    {sum(1 for d in result.details if d.get('gt_rating') is not None)}",
        "=" * 60,
    ]
    return "\n".join(lines)