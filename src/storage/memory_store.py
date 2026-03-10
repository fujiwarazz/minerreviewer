from __future__ import annotations

import logging
import uuid
from pathlib import Path

from common.types import ExperienceCard
from common.utils import read_json, write_json

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cards: list[ExperienceCard] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        data = read_json(self.path)
        self.cards = [ExperienceCard(**item) for item in data]

    def _save(self) -> None:
        write_json(self.path, [card.model_dump() for card in self.cards])

    def list_active(self, venue_id: str, theme: str) -> list[ExperienceCard]:
        return [card for card in self.cards if card.venue_id == venue_id and card.theme == theme and card.active]

    def add_or_update(self, venue_id: str, theme: str, content: str, quality: float, similarity_threshold: float, trace: dict) -> ExperienceCard:
        existing = self._find_similar(venue_id, theme, content, similarity_threshold)
        if existing:
            existing.active = False
            new_version = existing.version + 1
        else:
            new_version = 1
        card = ExperienceCard(
            card_id=str(uuid.uuid4()),
            venue_id=venue_id,
            theme=theme,
            content=content,
            version=new_version,
            active=True,
            quality=quality,
            source_trace=trace,
        )
        self.cards.append(card)
        self._save()
        logger.info("Stored memory card %s v%s", card.card_id, card.version)
        return card

    def rollback(self, card_id: str) -> None:
        target = next((card for card in self.cards if card.card_id == card_id), None)
        if target is None:
            raise ValueError("Card not found")
        target.active = False
        self._save()

    def _find_similar(self, venue_id: str, theme: str, content: str, threshold: float) -> ExperienceCard | None:
        tokens = set(content.lower().split())
        for card in self.cards:
            if card.venue_id != venue_id or card.theme != theme or not card.active:
                continue
            card_tokens = set(card.content.lower().split())
            if not tokens or not card_tokens:
                continue
            overlap = len(tokens & card_tokens) / max(len(tokens | card_tokens), 1)
            if overlap >= threshold:
                return card
        return None
