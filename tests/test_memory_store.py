from __future__ import annotations

from common.types import ExperienceCard, PaperCase
from pipeline.distill_experience import ExperienceDistiller
from storage.memory_store import MemoryStore
from storage.memory_registry import MemoryRegistry
from storage.multi_case_store import MultiCaseStore
from storage.multi_memory_store import MultiMemoryStore
from pipeline.memory_editor import MemoryEditor


def test_memory_versioning(tmp_path) -> None:
    store_path = tmp_path / "memory.json"
    store = MemoryStore(store_path)
    first = store.add_or_update("venue", "theme", "alpha beta", 0.5, 0.5, {})
    second = store.add_or_update("venue", "theme", "alpha beta gamma", 0.6, 0.2, {})

    assert first.version == 1
    assert second.version == 2
    assert not any(card.active and card.card_id == first.card_id for card in store.cards)
    assert any(card.active and card.card_id == second.card_id for card in store.cards)


def test_memory_editor_routes_to_matching_venue_year(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    registry = MemoryRegistry()
    registry.register_memory("iclr_2023", "ICLR", 2023, 0, 0)
    registry.register_memory("iclr_2024", "ICLR", 2024, 0, 0)

    memory_store = MultiMemoryStore(registry=registry)
    case_store = MultiCaseStore(registry=registry)
    editor = MemoryEditor(memory_store=memory_store, case_store=case_store)

    card = ExperienceCard(
        card_id="card-1",
        kind="policy",
        scope="venue",
        venue_id="ICLR",
        theme="quality",
        content="Strong empirical work should compare against competitive baselines.",
        utility=0.8,
        confidence=0.8,
        metadata={"memory_year": 2024, "memory_type": "generic_policy"},
    )
    case = PaperCase(
        case_id="case-1",
        paper_id="paper-1",
        venue_id="ICLR",
        year=2024,
        title="Test",
        abstract="Abstract",
    )

    assert editor.admit(card) == "admitted_long"
    assert editor.admit_paper_case(case) is True

    card_path_2023 = tmp_path / "data/processed/memory/iclr_2023/policy_cards.jsonl"
    assert (tmp_path / "data/processed/memory/iclr_2024/policy_cards.jsonl").exists()
    assert (tmp_path / "data/processed/memory/iclr_2024/cases.jsonl").exists()
    assert not card_path_2023.exists() or card_path_2023.read_text(encoding="utf-8").strip() == ""


def test_distiller_rejects_paper_specific_memory() -> None:
    distiller = ExperienceDistiller(llm=None)

    assert distiller._is_generalizable_memory(  # type: ignore[attr-defined]
        "The paper only reports gains in Table 3 and misses comparisons in Figure 2."
    ) is False
    assert distiller._is_generalizable_memory(  # type: ignore[attr-defined]
        "Empirical claims should compare against strong task-appropriate baselines."
    ) is True


def test_distiller_uses_domain_scope_when_signature_has_domain() -> None:
    distiller = ExperienceDistiller(llm=None)

    assert distiller._policy_scope(  # type: ignore[attr-defined]
        type("Sig", (), {"domain": "vision"})()
    ) == "domain"
