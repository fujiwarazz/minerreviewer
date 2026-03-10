from __future__ import annotations

from common.types import Criterion
from storage.memory_store import MemoryStore


def should_update_memory(raw_rating: float, calibrated: float, stable_margin: float, borderline_low: float, borderline_high: float) -> bool:
    stable = abs(calibrated - raw_rating) < stable_margin
    borderline = borderline_low <= calibrated <= borderline_high
    return stable or borderline


def update_memory(
    store: MemoryStore,
    venue_id: str,
    policy_criteria: list[Criterion],
    raw_rating: float,
    calibrated: float,
    similarity_threshold: float,
    stable_margin: float,
    borderline_low: float,
    borderline_high: float,
    trace: dict,
) -> list[str]:
    if not should_update_memory(raw_rating, calibrated, stable_margin, borderline_low, borderline_high):
        return []
    updated: list[str] = []
    for criterion in policy_criteria:
        card = store.add_or_update(
            venue_id=venue_id,
            theme=criterion.theme,
            content=criterion.text,
            quality=calibrated,
            similarity_threshold=similarity_threshold,
            trace=trace,
        )
        updated.append(card.card_id)
    return updated
