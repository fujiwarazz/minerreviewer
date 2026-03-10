from __future__ import annotations

import logging
import random
from typing import Callable

import numpy as np

from common.types import Criterion

logger = logging.getLogger(__name__)


def _default_embedder(texts: list[str], model_name: str) -> np.ndarray:
    raise RuntimeError("Default embedder not configured")


class CriteriaDistiller:
    def __init__(self, model_name: str, embedder: Callable[[list[str], str], np.ndarray] | None = None) -> None:
        self.model_name = model_name
        self.embedder = embedder or _default_embedder

    def dedup(self, criteria: list[Criterion], threshold: float) -> list[Criterion]:
        if not criteria:
            return []
        texts = [c.text for c in criteria]
        embeddings = self.embedder(texts, self.model_name)
        norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        keep: list[Criterion] = []
        keep_indices: list[int] = []
        for idx, criterion in enumerate(criteria):
            if any(float(np.dot(norms[idx], norms[j])) >= threshold for j in keep_indices):
                continue
            keep.append(criterion)
            keep_indices.append(idx)
        logger.info("Deduped %s -> %s criteria", len(criteria), len(keep))
        return keep

    def select(
        self,
        criteria: list[Criterion],
        max_total: int,
        max_per_theme: int,
        seed: int,
        strategy: str | None = None,
        epsilon: float = 1.0,
    ) -> list[Criterion]:
        strategy = strategy or "random"
        if strategy == "max_volume":
            selected = self._select_max_volume(criteria, max_total, max_per_theme, epsilon)
            logger.info("Selected %s criteria by max_volume", len(selected))
            return selected
        random.seed(seed)
        buckets: dict[str, list[Criterion]] = {}
        for criterion in criteria:
            buckets.setdefault(criterion.theme, []).append(criterion)
        selected: list[Criterion] = []
        for theme, items in buckets.items():
            random.shuffle(items)
            selected.extend(items[:max_per_theme])
        random.shuffle(selected)
        selected = selected[:max_total]
        logger.info("Selected %s criteria by random", len(selected))
        return selected

    def _select_max_volume(
        self,
        criteria: list[Criterion],
        max_total: int,
        max_per_theme: int,
        epsilon: float,
    ) -> list[Criterion]:
        if not criteria or max_total <= 0:
            return []
        texts = [c.text for c in criteria]
        embeddings = self.embedder(texts, self.model_name).astype("float64")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        theme_counts: dict[str, int] = {}
        selected_indices: list[int] = []

        def score(indices: list[int]) -> float:
            if not indices:
                return 0.0
            z = embeddings[indices]
            n, d = z.shape
            alpha = d / max(n * (epsilon ** 2), 1e-12)
            cov = z.T @ z
            matrix = np.eye(d) + alpha * cov
            sign, logdet = np.linalg.slogdet(matrix)
            if sign <= 0:
                return -float("inf")
            return 0.5 * logdet

        for _ in range(min(max_total, len(criteria))):
            best_idx = None
            best_score = -float("inf")
            for idx, criterion in enumerate(criteria):
                if idx in selected_indices:
                    continue
                if theme_counts.get(criterion.theme, 0) >= max_per_theme:
                    continue
                candidate_indices = selected_indices + [idx]
                candidate_score = score(candidate_indices)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_idx = idx
            if best_idx is None:
                break
            selected_indices.append(best_idx)
            theme = criteria[best_idx].theme
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        return [criteria[idx] for idx in selected_indices]
