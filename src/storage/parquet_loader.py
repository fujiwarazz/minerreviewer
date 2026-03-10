from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import json

from common.types import Paper, Review

logger = logging.getLogger(__name__)


def load_parquet_files(paths: list[str | Path], venue_id: str | None = None) -> tuple[list[Paper], list[Review]]:
    papers: dict[str, Paper] = {}
    reviews: list[Review] = []
    for path in paths:
        df = pd.read_parquet(path)
        for idx, row in df.iterrows():
            paper_id = str(row.get("paper_id") or row.get("id") or f"{Path(path).stem}_{idx}")
            venue = venue_id or str(row.get("venue") or row.get("venue_id") or "ICLR")
            year = _parse_int(row.get("year"))
            title = _to_text(row.get("title"))
            abstract = _to_text(row.get("abstract"))
            if paper_id not in papers:
                papers[paper_id] = Paper(
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    venue_id=venue,
                    year=year,
                    authors=_to_list(row.get("authors")),
                    fulltext=_to_text(row.get("fulltext") or row.get("full_text") or row.get("paper_content")),
                )
            decision = _to_text(row.get("decision"))
            for review_text, rating in _extract_reviews(row):
                review_id = f"{paper_id}__{len(reviews)}"
                reviews.append(
                    Review(
                        review_id=review_id,
                        paper_id=paper_id,
                        venue_id=venue,
                        year=year,
                        rating=rating,
                        text=review_text,
                        decision=decision,
                    )
                )
    logger.info("Loaded %s papers and %s reviews from parquet", len(papers), len(reviews))
    return list(papers.values()), reviews


def load_parquet_paper(path: str | Path, row_index: int = 0, venue_id: str | None = None) -> Paper:
    df = pd.read_parquet(path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index {row_index} out of range for {path}")
    row = df.iloc[row_index]
    paper_id = str(row.get("paper_id") or row.get("id") or f"{Path(path).stem}_{row_index}")
    venue = venue_id or str(row.get("venue") or row.get("venue_id") or "ICLR")
    year = _parse_int(row.get("year"))
    title = _to_text(row.get("title"))
    abstract = _to_text(row.get("abstract"))
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        venue_id=venue,
        year=year,
        authors=_to_list(row.get("authors")),
        fulltext=_to_text(row.get("fulltext") or row.get("full_text") or row.get("paper_content")),
    )


def load_parquet_ground_truth(path: str | Path, row_index: int = 0) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index {row_index} out of range for {path}")
    row = df.iloc[row_index]
    decision = row.get("decision")
    raw = row.get("reviews_json")
    if _is_empty(raw):
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    outputs: list[dict[str, Any]] = []
    for entry in parsed if isinstance(parsed, list) else []:
        content = entry.get("content") if isinstance(entry, dict) else None
        if not isinstance(content, dict):
            continue
        strengths = content.get("strengths")
        weaknesses = content.get("weaknesses")
        rating_value = None
        scores = entry.get("scores") if isinstance(entry.get("scores"), dict) else {}
        if isinstance(scores, dict):
            rating_value = scores.get("rating", {}).get("value") if isinstance(scores.get("rating"), dict) else None
        outputs.append(
            {
                "reply_id": entry.get("reply_id"),
                "strengths": strengths,
                "weaknesses": weaknesses,
                "rating": rating_value,
                "confidence": scores.get("confidence", {}).get("value") if isinstance(scores.get("confidence"), dict) else None,
                "decision": decision,
            }
        )
    return outputs


def _extract_reviews(row: pd.Series) -> list[tuple[str, float | None]]:
    candidates = [
        row.get("review"),
        row.get("reviews"),
        row.get("summary_review"),
        row.get("meta_review"),
        row.get("review_text"),
        row.get("reviews_json"),
    ]
    outputs: list[tuple[str, float | None]] = []
    for item in candidates:
        if _is_empty(item):
            continue
        if isinstance(item, str) and item.strip().startswith("[") and "review" in item:
            try:
                parsed = json.loads(item)
                if isinstance(parsed, list):
                    for entry in parsed:
                        text, rating = _extract_review_entry(entry)
                        if text:
                            outputs.append((text, rating))
                    continue
            except json.JSONDecodeError:
                pass
        if isinstance(item, list):
            for entry in item:
                text, rating = _extract_review_entry(entry)
                if text:
                    outputs.append((text, rating))
        else:
            text, rating = _extract_review_entry(item)
            if text:
                outputs.append((text, rating))
    return outputs


def _extract_review_entry(entry: Any) -> tuple[str, float | None]:
    if isinstance(entry, dict):
        text_parts = []
        content = entry.get("content") if "content" in entry else entry
        for key in ("summary", "strengths", "weaknesses", "review", "comment", "title"):
            value = content.get(key) if isinstance(content, dict) else None
            if not _is_empty(value):
                text_parts.append(str(value))
        text = "\n".join(text_parts)
        rating_value = entry.get("rating") or entry.get("score")
        scores = entry.get("scores") if isinstance(entry.get("scores"), dict) else {}
        if not rating_value and isinstance(scores, dict):
            rating_value = scores.get("rating", {}).get("value") if isinstance(scores.get("rating"), dict) else None
        rating = _parse_rating(rating_value)
        return text.strip(), rating
    if _is_empty(entry):
        return "", None
    return str(entry).strip(), None


def _to_text(value: Any) -> str:
    if _is_empty(value):
        return ""
    return str(value)


def _to_list(value: Any) -> list[str]:
    if _is_empty(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_rating(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parts = value.split(":")
        try:
            return float(parts[0])
        except ValueError:
            return None
    return None


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return pd.isna(value)
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False
