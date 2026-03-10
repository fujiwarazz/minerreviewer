from __future__ import annotations

import numpy as np

from common.types import Criterion
from pipeline.distill_criteria import CriteriaDistiller


def test_dedup_selection() -> None:
    def fake_embedder(texts: list[str], model_name: str) -> np.ndarray:
        mapping = {
            "a": [1.0, 0.0],
            "a_dup": [0.99, 0.01],
            "b": [0.0, 1.0],
        }
        return np.array([mapping[t] for t in texts])

    criteria = [
        Criterion(criterion_id="1", text="a", theme="t1", kind="content"),
        Criterion(criterion_id="2", text="a_dup", theme="t1", kind="content"),
        Criterion(criterion_id="3", text="b", theme="t2", kind="content"),
    ]
    distiller = CriteriaDistiller("fake", embedder=fake_embedder)
    deduped = distiller.dedup(criteria, threshold=0.95)
    assert len(deduped) == 2
    selected = distiller.select(deduped, max_total=2, max_per_theme=1, seed=1)
    assert len(selected) == 2
