"""MultiVectorMemoryStore: 聚合多个VectorMemoryStore

支持：
1. 从多个活跃记忆库加载记忆
2. 按agent分组检索
3. 热插拔式管理
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from clients.embedding_client import EmbeddingClient
from common.types import ExperienceCard
from storage.vector_memory_store import VectorMemoryStore
from storage.memory_registry import MemoryRegistry
from storage.milvus_store import MilvusConfig

logger = logging.getLogger(__name__)


class MultiVectorMemoryStore:
    """管理多个 VectorMemoryStore，支持热插拔"""

    def __init__(
        self,
        registry: MemoryRegistry | None = None,
        embedding_client: EmbeddingClient | None = None,
        milvus_config: MilvusConfig | None = None,
        memory_root: str | Path = "data/processed/memory",
    ) -> None:
        self.registry = registry or MemoryRegistry()
        self.embedding_client = embedding_client
        self.milvus_config = milvus_config
        self.memory_root = Path(memory_root)
        self._stores: dict[str, VectorMemoryStore] = {}

        self._load_active_stores()

    def _load_active_stores(self) -> None:
        """加载所有活跃记忆库"""
        for memory_id in self.registry.get_active_memories():
            memory_path = self.memory_root / memory_id
            cards_path = memory_path / "agent_memories.jsonl"

            if cards_path.exists():
                store = VectorMemoryStore(
                    cards_path,
                    embedding_client=self.embedding_client,
                    milvus_config=self.milvus_config,
                )
                self._stores[memory_id] = store
                logger.info(
                    "Loaded VectorMemoryStore '%s' with %d cards",
                    memory_id,
                    len(store.cards),
                )

    def add_store(self, memory_id: str, path: str | Path) -> None:
        """动态添加记忆库"""
        store = VectorMemoryStore(
            path,
            embedding_client=self.embedding_client,
            milvus_config=self.milvus_config,
        )
        self._stores[memory_id] = store
        logger.info("Added VectorMemoryStore '%s' with %d cards", memory_id, len(store.cards))

    def remove_store(self, memory_id: str) -> bool:
        """移除记忆库"""
        if memory_id in self._stores:
            del self._stores[memory_id]
            logger.info("Removed VectorMemoryStore '%s'", memory_id)
            return True
        return False

    def retrieve_for_agent(
        self,
        query_text: str,
        agent_name: str,
        kind: Literal["strength", "critique", "failure"] | None = None,
        top_k: int = 10,
        merge_stores: bool = True,
        primary_area: str | None = None,  # 新增：按领域过滤
    ) -> list[tuple[ExperienceCard, dict[str, float]]]:
        """为特定agent检索记忆

        Args:
            query_text: 查询文本
            agent_name: agent名称（如 theme_quality, arbiter）
            kind: 过滤卡片类型
            top_k: 返回数量
            merge_stores: 是否合并多个store的结果

        Returns:
            list of (card, scores_dict)
        """
        all_results = []

        for store_id, store in self._stores.items():
            results = store.retrieve_cards(
                query_text=query_text,
                owner_agent=agent_name,
                kind=kind,
                top_k=top_k,
                primary_area=primary_area,
            )
            # 添加store_id到scores
            for card, scores in results:
                scores["store_id"] = store_id
                all_results.append((card, scores))

        if merge_stores:
            # 合并并重排序
            all_results.sort(key=lambda x: x[1]["final_score"], reverse=True)
            return all_results[:top_k]

        return all_results

    def retrieve_all(
        self,
        query_text: str,
        top_k: int = 20,
        kind: Literal["strength", "critique", "failure"] | None = None,
        primary_area: str | None = None,  # 新增：按领域过滤
    ) -> list[tuple[ExperienceCard, dict[str, float]]]:
        """检索所有记忆（不限agent）

        Args:
            query_text: 查询文本
            top_k: 返回数量
            kind: 过滤卡片类型

        Returns:
            list of (card, scores_dict)
        """
        all_results = []

        for store_id, store in self._stores.items():
            results = store.retrieve_cards(
                query_text=query_text,
                kind=kind,
                top_k=top_k,
            )
            for card, scores in results:
                scores["store_id"] = store_id
                all_results.append((card, scores))

        all_results.sort(key=lambda x: x[1]["final_score"], reverse=True)
        return all_results[:top_k]

    def retrieve_for_agents(
        self,
        query_text: str,
        agent_names: list[str],
        top_k_per_agent: int = 8,
        primary_area: str | None = None,  # 新增：按领域过滤
    ) -> dict[str, list[tuple[ExperienceCard, dict[str, float]]]]:
        """为多个agent检索记忆

        Args:
            query_text: 查询文本
            agent_names: agent名称列表
            top_k_per_agent: 每个agent的返回数量

        Returns:
            dict mapping agent_name to list of (card, scores)
        """
        results = {}
        for agent_name in agent_names:
            agent_results = self.retrieve_for_agent(
                query_text=query_text,
                agent_name=agent_name,
                top_k=top_k_per_agent,
                primary_area=primary_area,
            )
            results[agent_name] = agent_results
            logger.info(
                "Retrieved %d memories for agent '%s'",
                len(agent_results),
                agent_name,
            )
        return results

    def add_card_to_store(
        self,
        memory_id: str,
        card: ExperienceCard,
        owner_agent: str | None = None,
    ) -> str | None:
        """向指定store添加卡片"""
        store = self._stores.get(memory_id)
        if store:
            card_id = store.add_card(card, owner_agent=owner_agent)
            return card_id
        logger.warning("Store '%s' not found", memory_id)
        return None

    def batch_add_cards_to_store(
        self,
        memory_id: str,
        cards: list[ExperienceCard],
        owner_agent: str | None = None,
    ) -> list[str]:
        """向指定store批量添加卡片"""
        store = self._stores.get(memory_id)
        if store:
            return store.batch_add_cards(cards, owner_agent=owner_agent)
        logger.warning("Store '%s' not found", memory_id)
        return []

    def get_store(self, memory_id: str) -> VectorMemoryStore | None:
        """获取指定store"""
        return self._stores.get(memory_id)

    def get_all_cards(self) -> list[ExperienceCard]:
        """获取所有卡片"""
        all_cards = []
        for store in self._stores.values():
            all_cards.extend(store.cards)
        return all_cards

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        stats = {
            "num_stores": len(self._stores),
            "stores": {},
            "total_cards": 0,
        }

        for store_id, store in self._stores.items():
            store_stats = store.get_stats()
            stats["stores"][store_id] = store_stats
            stats["total_cards"] += store_stats["total_cards"]

        return stats