from __future__ import annotations

from storage.memory_store import MemoryStore


def test_memory_versioning(tmp_path) -> None:
    store_path = tmp_path / "memory.json"
    store = MemoryStore(store_path)
    first = store.add_or_update("venue", "theme", "alpha beta", 0.5, 0.5, {})
    second = store.add_or_update("venue", "theme", "alpha beta gamma", 0.6, 0.2, {})

    assert first.version == 1
    assert second.version == 2
    assert not any(card.active and card.card_id == first.card_id for card in store.cards)
    assert any(card.active and card.card_id == second.card_id for card in store.cards)
