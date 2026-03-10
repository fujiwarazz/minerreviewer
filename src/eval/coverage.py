from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from clients.embedding_client import EmbeddingClient
from clients.llm_client import LLMClient

logger = logging.getLogger(__name__)


def evaluate_coverage(
    strengths: list[str],
    weaknesses: list[str],
    ground_truth_reviews: list[dict[str, Any]],
    llm: LLMClient,
    embedding_client: EmbeddingClient,
    config: dict[str, Any],
) -> dict[str, Any]:
    method = config.get("method", "llm")
    max_points = int(config.get("max_points", 12))
    threshold = float(config.get("threshold", 0.55))

    gt_strengths = _extract_gt_points(ground_truth_reviews, "strengths", max_points)
    gt_weaknesses = _extract_gt_points(ground_truth_reviews, "weaknesses", max_points)

    results: dict[str, Any] = {"method": method}
    results["strengths"] = _score_aspect(
        strengths,
        gt_strengths,
        llm,
        embedding_client,
        method,
        threshold,
        "strengths",
    )
    results["weaknesses"] = _score_aspect(
        weaknesses,
        gt_weaknesses,
        llm,
        embedding_client,
        method,
        threshold,
        "weaknesses",
    )
    return results


def _score_aspect(
    preds: list[str],
    gt_points: list[str],
    llm: LLMClient,
    embedding_client: EmbeddingClient,
    method: str,
    threshold: float,
    aspect: str,
) -> dict[str, Any]:
    if not gt_points:
        return {"score": 1.0, "explanation": "no_ground_truth"}
    if not preds:
        return {"score": 0.0, "explanation": "empty_predictions"}
    if method == "embedding":
        return _embedding_coverage(preds, gt_points, embedding_client, threshold)
    return _llm_coverage(preds, gt_points, llm, aspect)


def _embedding_coverage(preds: list[str], gt_points: list[str], embedding_client: EmbeddingClient, threshold: float) -> dict[str, Any]:
    gt_embs = embedding_client.embed(gt_points)
    pred_embs = embedding_client.embed(preds)
    if gt_embs.size == 0 or pred_embs.size == 0:
        return {"score": 0.0, "explanation": "embedding_failed"}
    gt_norm = gt_embs / (np.linalg.norm(gt_embs, axis=1, keepdims=True) + 1e-9)
    pred_norm = pred_embs / (np.linalg.norm(pred_embs, axis=1, keepdims=True) + 1e-9)
    sim = np.dot(gt_norm, pred_norm.T)
    best = np.max(sim, axis=1)
    valid = [float(score) if score > threshold else 0.0 for score in best]
    unmatched = [gt_points[idx] for idx, score in enumerate(best) if score <= threshold]
    return {"score": round(float(np.mean(valid)), 3), "explanation": "", "unmatched_points": unmatched}


def _llm_coverage(preds: list[str], gt_points: list[str], llm: LLMClient, aspect: str) -> dict[str, Any]:
    gt_list = "\n".join([f"{i + 1}. {item}" for i, item in enumerate(gt_points)])
    pred_list = "\n".join([f"{i + 1}. {item}" for i, item in enumerate(preds)])
    prompt = "\n".join(
        [
            "You are a strict evaluator. Determine whether each GT item is covered by Pred items (semantic match).",
            f"Aspect: {aspect}",
            f"Ground Truth List:\n{gt_list}",
            f"Generated List:\n{pred_list}",
            "Return JSON: {matches: [{gt_id, matched, matched_pred_id, reasoning}], summary: {total_gt_items, total_matched, coverage_score}}",
        ]
    )
    try:
        response = llm.generate_json(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM coverage failed: %s", exc)
        return {"score": 0.0, "explanation": "llm_failed"}
    summary = response.get("summary", {}) if isinstance(response, dict) else {}
    score = summary.get("coverage_score")
    if score is None:
        matches = response.get("matches", []) if isinstance(response, dict) else []
        total = summary.get("total_gt_items") or len(gt_points)
        matched = sum(1 for item in matches if item.get("matched"))
        score = matched / total if total else 0.0
    unmatched: list[str] = []
    matches = response.get("matches", []) if isinstance(response, dict) else []
    for item in matches:
        if not item.get("matched"):
            gt_id = item.get("gt_id")
            if isinstance(gt_id, int) and 1 <= gt_id <= len(gt_points):
                unmatched.append(gt_points[gt_id - 1])
    return {
        "score": round(float(score), 3),
        "explanation": json.dumps(response, ensure_ascii=True),
        "unmatched_points": unmatched,
    }


def _extract_gt_points(ground_truth_reviews: list[dict[str, Any]], field: str, max_points: int) -> list[str]:
    points: list[str] = []
    for review in ground_truth_reviews:
        value = review.get(field)
        if isinstance(value, dict):
            value = value.get("value")
        if value:
            points.extend(_split_points(str(value)))
    return points[:max_points]


def _split_points(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("@@") if p and p.strip()]
    if not parts:
        tmp: list[str] = []
        for line in text.splitlines():
            line = line.strip(" -*\t")
            if line:
                tmp.append(line)
        parts = tmp or [text]
    return parts
