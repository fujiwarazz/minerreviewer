"""MultiMemoryStore: 聚合多个记忆库的 MemoryStore

支持：
1. 从多个活跃记忆库加载 ExperienceCards
2. 聚合检索和过滤
3. 热插拔式管理
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from common.types import ExperienceCard
from storage.memory_store import MemoryStore
from storage.memory_registry import MemoryRegistry, MEMORY_ROOT

logger = logging.getLogger(__name__)


class MultiMemoryStore:
    """管理多个 MemoryStore，支持热插拔记忆库"""

    def __init__(self, registry: MemoryRegistry | None = None, skip_load: bool = False) -> None:
        self.registry = registry or MemoryRegistry()
        self._stores: dict[str, MemoryStore] = {}
        self._cards_cache: list[ExperienceCard] | None = None  # 缓存聚合后的 cards

        # skip_load=True 时不从文件加载，用于训练时从空memory开始
        if not skip_load:
            self._load_active_stores()

    @property
    def cards(self) -> list[ExperienceCard]:
        """向后兼容：聚合所有活跃记忆库的 cards"""
        if self._cards_cache is None:
            self._cards_cache = self.list_cards()
        return self._cards_cache

    def _load_active_stores(self) -> None:
        """加载所有活跃记忆库"""
        for memory_id in self.registry.get_active_memories():
            memory_path = self.registry.get_memory_path(memory_id)
            if memory_path:
                cards_path = memory_path / "policy_cards.jsonl"
                store = MemoryStore(cards_path)
                self._annotate_store_cards(memory_id, store)
                self._stores[memory_id] = store
                logger.info("Loaded %d cards from %s", len(store.cards), memory_id)

    def refresh(self) -> None:
        """刷新记忆库（重新加载活跃记忆库）"""
        self._stores.clear()
        self._cards_cache = None
        self._load_active_stores()

    def activate_memory(self, memory_id: str) -> bool:
        """激活记忆库并加载"""
        if self.registry.activate_memory(memory_id):
            memory_path = self.registry.get_memory_path(memory_id)
            if memory_path:
                cards_path = memory_path / "policy_cards.jsonl"
                store = MemoryStore(cards_path)
                self._annotate_store_cards(memory_id, store)
                self._stores[memory_id] = store
                self._cards_cache = None  # 清空缓存
                logger.info("Activated and loaded %s (%d cards)", memory_id, len(store.cards))
                return True
        return False

    def deactivate_memory(self, memory_id: str) -> bool:
        """停用记忆库并卸载"""
        if self.registry.deactivate_memory(memory_id):
            if memory_id in self._stores:
                del self._stores[memory_id]
                self._cards_cache = None  # 清空缓存
                logger.info("Deactivated and unloaded %s", memory_id)
            return True
        return False

    def list_active(
        self,
        venue_id: str | None = None,
        theme: str | None = None,
        kind: Literal["strength", "critique", "failure"] | None = None,
        scope: Literal["global", "venue", "paper_type", "domain"] | None = None,
    ) -> list[ExperienceCard]:
        """列出所有活跃记忆库中的卡片（向后兼容）"""
        return self.list_cards(venue_id=venue_id, theme=theme, kind=kind, scope=scope)

    def list_cards(
        self,
        venue_id: str | None = None,
        theme: str | None = None,
        kind: Literal["strength", "critique", "failure"] | None = None,
        scope: Literal["global", "venue", "paper_type", "domain"] | None = None,
    ) -> list[ExperienceCard]:
        """列出所有活跃记忆库中的卡片"""
        all_cards = []
        for store in self._stores.values():
            cards = store.list_active(venue_id=venue_id, theme=theme, kind=kind, scope=scope)
            all_cards.extend(cards)
        return all_cards

    def list_by_kind(
        self,
        kind: Literal["strength", "critique", "failure"],
        venue_id: str | None = None,
    ) -> list[ExperienceCard]:
        """按类型列出卡片"""
        all_cards = []
        for store in self._stores.values():
            cards = store.list_by_kind(kind, venue_id=venue_id)
            all_cards.extend(cards)
        return all_cards

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        stats = {
            "active_memories": len(self._stores),
            "total_cards": sum(len(s.cards) for s in self._stores.values()),
            "by_memory": {},
        }
        for memory_id, store in self._stores.items():
            info = self.registry.get_memory_info(memory_id)
            stats["by_memory"][memory_id] = {
                "venue": info.get("venue") if info else None,
                "year": info.get("year") if info else None,
                "cards_count": len(store.cards),
            }
        return stats

    def get_cards_for_venue(
        self,
        venue_id: str,
        kind: Literal["strength", "critique", "failure"] = "strength",
        top_k: int = 50,
    ) -> list[ExperienceCard]:
        """获取指定 venue 的卡片"""
        cards = self.list_by_kind(kind, venue_id=venue_id)
        # 按 utility/confidence 排序
        cards.sort(key=lambda c: (c.utility or 0.5) + (c.confidence or 0.5), reverse=True)
        return cards[:top_k]

    def get_cards_for_theme(
        self,
        theme: str,
        venue_id: str | None = None,
        top_k: int = 20,
    ) -> list[ExperienceCard]:
        """获取指定 theme 的卡片"""
        cards = self.list_cards(venue_id=venue_id, theme=theme, kind="strength")
        cards.sort(key=lambda c: (c.utility or 0.5) + (c.confidence or 0.5), reverse=True)
        return cards[:top_k]

    def get_store_for_venue_year(self, venue_id: str | None, year: int | None) -> MemoryStore | None:
        """按会议和年份查找底层 store，用于精确写回路由"""
        if venue_id is None or year is None:
            return None

        memory_id = self.registry.get_memory_for_venue_year(venue_id, year)
        if memory_id is None:
            return None

        store = self._stores.get(memory_id)
        if store is not None:
            return store

        if self.activate_memory(memory_id):
            return self._stores.get(memory_id)
        return None

    def _annotate_store_cards(self, memory_id: str, store: MemoryStore) -> None:
        """为聚合读取的 cards 补充来源记忆库信息"""
        info = self.registry.get_memory_info(memory_id) or {}
        for card in store.cards:
            card.metadata.setdefault("_memory_id", memory_id)
            card.metadata.setdefault("_memory_year", info.get("year"))
            card.metadata.setdefault("_memory_venue", info.get("venue"))
