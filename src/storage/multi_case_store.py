"""MultiCaseStore: 聚合多个记忆库的 CaseStore

支持：
1. 从多个活跃记忆库加载 cases
2. 聚合检索结果
3. 热插拔式管理
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from clients.embedding_client import EmbeddingClient
from common.types import PaperCase, PaperSignature
from storage.case_store import CaseStore
from storage.memory_registry import MemoryRegistry, MEMORY_ROOT
from storage.milvus_store import MilvusConfig

logger = logging.getLogger(__name__)


class MultiCaseStore:
    """管理多个 CaseStore，支持热插拔记忆库"""

    def __init__(
        self,
        registry: MemoryRegistry | None = None,
        embedding_client: EmbeddingClient | None = None,
        milvus_config: MilvusConfig | None = None,
        embedding_weight: float = 0.5,
        signature_weight: float = 0.4,
        venue_match_bonus: float = 0.1,
        skip_load: bool = False,  # 不从文件加载，用于训练时从空memory开始
    ) -> None:
        self.registry = registry or MemoryRegistry()
        self.embedding_client = embedding_client
        self.milvus_config = milvus_config
        self.embedding_weight = embedding_weight
        self.signature_weight = signature_weight
        self.venue_match_bonus = venue_match_bonus

        # 按记忆库 ID 存储 CaseStore
        self._stores: dict[str, CaseStore] = {}

        # skip_load=True 时不从文件加载
        if not skip_load:
            self._load_active_stores()

    def _load_active_stores(self) -> None:
        """加载所有活跃记忆库"""
        for memory_id in self.registry.get_active_memories():
            memory_path = self.registry.get_memory_path(memory_id)
            if memory_path:
                cases_path = memory_path / "cases.jsonl"
                store = CaseStore(
                    path=cases_path,
                    embedding_client=self.embedding_client,
                    milvus_config=self.milvus_config,
                    embedding_weight=self.embedding_weight,
                    signature_weight=self.signature_weight,
                    venue_match_bonus=self.venue_match_bonus,
                )
                self._stores[memory_id] = store
                logger.info("Loaded %d cases from %s", len(store.cases), memory_id)

    def refresh(self) -> None:
        """刷新记忆库（重新加载活跃记忆库）"""
        self._stores.clear()
        self._load_active_stores()

    def activate_memory(self, memory_id: str) -> bool:
        """激活记忆库并加载"""
        if self.registry.activate_memory(memory_id):
            memory_path = self.registry.get_memory_path(memory_id)
            if memory_path:
                cases_path = memory_path / "cases.jsonl"
                store = CaseStore(
                    path=cases_path,
                    embedding_client=self.embedding_client,
                    milvus_config=self.milvus_config,
                    embedding_weight=self.embedding_weight,
                    signature_weight=self.signature_weight,
                    venue_match_bonus=self.venue_match_bonus,
                )
                self._stores[memory_id] = store
                logger.info("Activated and loaded %s (%d cases)", memory_id, len(store.cases))
                return True
        return False

    def deactivate_memory(self, memory_id: str) -> bool:
        """停用记忆库并卸载"""
        if self.registry.deactivate_memory(memory_id):
            if memory_id in self._stores:
                del self._stores[memory_id]
                logger.info("Deactivated and unloaded %s", memory_id)
            return True
        return False

    def list_cases(self, venue_id: str | None = None, year: int | None = None) -> list[PaperCase]:
        """列出所有活跃记忆库中的 cases"""
        all_cases = []
        for store in self._stores.values():
            cases = store.list_cases(venue_id=venue_id, year=year)
            all_cases.extend(cases)
        return all_cases

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        stats = {
            "active_memories": len(self._stores),
            "total_cases": sum(len(s.cases) for s in self._stores.values()),
            "by_memory": {},
        }
        for memory_id, store in self._stores.items():
            info = self.registry.get_memory_info(memory_id)
            stats["by_memory"][memory_id] = {
                "venue": info.get("venue") if info else None,
                "year": info.get("year") if info else None,
                "cases_count": len(store.cases),
            }
        return stats

    def retrieve_cases(
        self,
        query_text: str,
        signature: PaperSignature | None = None,
        top_k: int = 10,
        venue_id: str | None = None,
        use_hybrid: bool = True,
        diversity_threshold: float = 0.9,
        exclude_paper_id: str | None = None,
        before_year: int | None = None,
    ) -> list[tuple[PaperCase, dict[str, float]]]:
        """从所有活跃记忆库检索相似案例

        策略：
        1. 如果指定了 venue_id，优先从该 venue 的记忆库检索
        2. 融合所有记忆库的结果
        3. 去重并排序
        """
        # 收集所有结果
        all_results: list[tuple[PaperCase, dict[str, float]]] = []

        # 确定要使用的记忆库
        target_memories = []
        if venue_id:
            target_memories = self.registry.get_memories_for_venue(venue_id, before_year)

        # 如果没有找到对应 venue 的记忆库，使用所有活跃记忆库
        if not target_memories:
            target_memories = self.registry.get_memories_for_year(before_year or 9999)

        for memory_id in target_memories:
            store = self._stores.get(memory_id)
            if store:
                results = store.retrieve_cases(
                    query_text=query_text,
                    signature=signature,
                    top_k=top_k * 2,  # 多取一些以便融合后去重
                    venue_id=venue_id,
                    use_hybrid=use_hybrid,
                    diversity_threshold=diversity_threshold,
                    exclude_paper_id=exclude_paper_id,
                    before_year=before_year,
                )
                all_results.extend(results)

        # 去重（按 paper_id）
        seen_paper_ids: set[str] = set()
        unique_results: list[tuple[PaperCase, dict[str, float]]] = []
        for case, scores in all_results:
            if case.paper_id and case.paper_id in seen_paper_ids:
                continue
            if case.paper_id:
                seen_paper_ids.add(case.paper_id)
            unique_results.append((case, scores))

        # 按分数排序
        unique_results.sort(key=lambda x: x[1].get("final_score", 0), reverse=True)

        # 应用多样性控制
        final_results = self._apply_diversity(unique_results, diversity_threshold)

        return final_results[:top_k]

    def _apply_diversity(
        self,
        results: list[tuple[PaperCase, dict[str, float]]],
        threshold: float,
    ) -> list[tuple[PaperCase, dict[str, float]]]:
        """多样性控制"""
        seen_papers: set[str] = set()
        venue_counts: dict[str, int] = {}
        year_counts: dict[int, int] = {}

        diverse: list[tuple[PaperCase, dict[str, float]]] = []
        max_per_venue = max(3, len(results) // 3)
        max_per_year = max(3, len(results) // 3)

        for case, scores in results:
            if case.paper_id and case.paper_id in seen_papers:
                continue

            venue = case.venue_id or "unknown"
            if venue_counts.get(venue, 0) >= max_per_venue:
                continue

            if case.year and year_counts.get(case.year, 0) >= max_per_year:
                continue

            if case.paper_id:
                seen_papers.add(case.paper_id)
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
            if case.year:
                year_counts[case.year] = year_counts.get(case.year, 0) + 1

            diverse.append((case, scores))

        return diverse

    def search_similar_cases(
        self,
        query_text: str,
        top_k: int = 10,
        venue_id: str | None = None,
        threshold: float = 0.0,
    ) -> list[PaperCase]:
        """向后兼容的搜索方法"""
        results = self.retrieve_cases(
            query_text=query_text,
            signature=None,
            top_k=top_k,
            venue_id=venue_id,
            use_hybrid=False,
        )
        return [case for case, scores in results if scores.get("final_score", 0) >= threshold]

    def get_store_for_venue_year(self, venue_id: str | None, year: int | None) -> CaseStore | None:
        """按会议和年份查找底层 case store，用于精确写回路由"""
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
