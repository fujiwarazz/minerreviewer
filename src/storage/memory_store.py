from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Literal

from common.types import ExperienceCard
from common.utils import read_json, write_json

logger = logging.getLogger(__name__)


class MemoryStore:
    """经验卡片存储，支持多种类型：policy/case/critique/failure"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cards: list[ExperienceCard] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        """加载卡片，支持 JSON 数组和 JSONL 格式"""
        import json
        with open(self.path, encoding="utf-8") as f:
            content = f.read().strip()

        # 支持 JSON 数组或 JSONL 格式
        if content.startswith('['):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split('\n') if line.strip()]

        self.cards = [ExperienceCard(**item) for item in data]

    def _save(self) -> None:
        write_json(self.path, [card.model_dump() for card in self.cards])

    def list_active(
        self,
        venue_id: str | None = None,
        theme: str | None = None,
        kind: Literal["policy", "case", "critique", "failure"] | None = None,
    ) -> list[ExperienceCard]:
        """列出活跃的卡片，支持多条件过滤"""
        cards = [card for card in self.cards if card.active]
        if venue_id is not None:
            cards = [card for card in cards if card.venue_id == venue_id]
        if theme is not None:
            cards = [card for card in cards if card.theme == theme]
        if kind is not None:
            cards = [card for card in cards if card.kind == kind]
        return cards

    def list_by_kind(
        self,
        kind: Literal["policy", "case", "critique", "failure"],
        venue_id: str | None = None,
    ) -> list[ExperienceCard]:
        """按类型列出卡片"""
        cards = [card for card in self.cards if card.kind == kind and card.active]
        if venue_id:
            cards = [card for card in cards if card.venue_id == venue_id]
        return cards

    def add_card(self, card: ExperienceCard) -> ExperienceCard:
        """添加新卡片"""
        if not card.card_id:
            card.card_id = str(uuid.uuid4())
        self.cards.append(card)
        self._save()
        logger.info("Added memory card %s (kind=%s)", card.card_id[:8], card.kind)
        return card

    def add_or_update(self, venue_id: str, theme: str, content: str, quality: float, similarity_threshold: float, trace: dict) -> ExperienceCard:
        """向后兼容的方法"""
        existing = self._find_similar(venue_id, theme, content, similarity_threshold)
        if existing:
            existing.active = False
            new_version = existing.version + 1
        else:
            new_version = 1
        card = ExperienceCard(
            card_id=str(uuid.uuid4()),
            kind="policy",
            scope="venue",
            venue_id=venue_id,
            theme=theme,
            content=content,
            version=new_version,
            active=True,
            utility=quality,
            confidence=quality,
            source_trace=trace,
        )
        self.cards.append(card)
        self._save()
        logger.info("Stored memory card %s v%s", card.card_id, card.version)
        return card

    def rollback(self, card_id: str) -> None:
        """回滚卡片"""
        target = next((card for card in self.cards if card.card_id == card_id), None)
        if target is None:
            raise ValueError("Card not found")
        target.active = False
        self._save()

    def find_similar(
        self,
        venue_id: str | None,
        theme: str,
        content: str,
        threshold: float,
    ) -> ExperienceCard | None:
        """查找相似卡片（公开方法）"""
        return self._find_similar(venue_id or "", theme, content, threshold)

    def _find_similar(self, venue_id: str, theme: str, content: str, threshold: float) -> ExperienceCard | None:
        """内部查找相似卡片"""
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

    def get_card(self, card_id: str) -> ExperienceCard | None:
        """根据 ID 获取卡片"""
        return next((card for card in self.cards if card.card_id == card_id), None)

    def update_card(self, card_id: str, updates: dict) -> None:
        """更新卡片"""
        card = self.get_card(card_id)
        if card:
            for key, value in updates.items():
                if hasattr(card, key):
                    setattr(card, key, value)
            self._save()

    def delete_card(self, card_id: str) -> bool:
        """删除卡片（软删除）"""
        card = self.get_card(card_id)
        if card:
            card.active = False
            self._save()
            return True
        return False

    def clear_inactive(self) -> int:
        """清理所有非活跃的卡片"""
        original_count = len(self.cards)
        self.cards = [card for card in self.cards if card.active]
        removed = original_count - len(self.cards)
        if removed > 0:
            self._save()
            logger.info("Removed %d inactive cards", removed)
        return removed
