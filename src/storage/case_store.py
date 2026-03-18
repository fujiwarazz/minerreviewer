"""CaseStore: 论文案例存储和检索，支持 hybrid retrieval"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from clients.embedding_client import EmbeddingClient
from common.types import Paper, PaperCase, PaperSignature, Review
from common.utils import write_json

logger = logging.getLogger(__name__)


class CaseStore:
    """管理 PaperCase 的存储和检索，支持 embedding + signature hybrid retrieval"""

    def __init__(
        self,
        path: str | Path,
        embedding_client: EmbeddingClient | None = None,
        embedding_weight: float = 0.5,
        signature_weight: float = 0.4,
        venue_match_bonus: float = 0.1,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_client = embedding_client
        self.embedding_weight = embedding_weight
        self.signature_weight = signature_weight
        self.venue_match_bonus = venue_match_bonus
        self.cases: list[PaperCase] = []
        self._index: dict[str, PaperCase] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.cases = [PaperCase(**item) for item in data]
        self._index = {case.case_id: case for case in self.cases}
        logger.info("Loaded %d cases from %s", len(self.cases), self.path)

    def _save(self) -> None:
        write_json(self.path, [case.model_dump() for case in self.cases])

    @staticmethod
    def get_case_text(case: PaperCase) -> str:
        """
        获取案例的文本表示，用于 embedding

        包含：title + abstract + decisive_issues + transferable_criteria
        这样比只用 title + abstract 更有信息量
        """
        parts = [case.title, case.abstract]
        if case.decisive_issues:
            parts.append("Key issues: " + "; ".join(case.decisive_issues[:3]))
        if case.transferable_criteria:
            parts.append("Criteria: " + "; ".join(case.transferable_criteria[:3]))
        return "\n".join(parts)

    def add_case(self, case: PaperCase) -> str:
        """添加案例，返回 case_id"""
        if not case.case_id:
            case.case_id = str(uuid.uuid4())
        if self.embedding_client and not case.embedding:
            # 使用增强的 case_text 而不是只用 title + abstract
            text = self.get_case_text(case)
            case.embedding = self.embedding_client.embed([text])[0].tolist()
        self._index[case.case_id] = case
        self.cases.append(case)
        self._save()
        logger.info("Added case %s", case.case_id)
        return case.case_id

    def get_case(self, case_id: str) -> PaperCase | None:
        """根据 ID 获取案例"""
        return self._index.get(case_id)

    def list_cases(self, venue_id: str | None = None, year: int | None = None) -> list[PaperCase]:
        """列出案例，可按 venue 和 year 过滤"""
        cases = self.cases
        if venue_id:
            cases = [c for c in cases if c.venue_id == venue_id]
        if year:
            cases = [c for c in cases if c.year is not None and c.year < year]
        return cases

    def retrieve_cases(
        self,
        query_text: str,
        signature: PaperSignature | None = None,
        top_k: int = 10,
        venue_id: str | None = None,
        use_hybrid: bool = True,
        diversity_threshold: float = 0.9,
    ) -> list[tuple[PaperCase, dict[str, float]]]:
        """
        统一 hybrid retrieval 入口

        Args:
            query_text: 查询文本
            signature: 论文结构化签名
            top_k: 返回数量
            venue_id: 目标 venue
            use_hybrid: 是否使用 hybrid (embedding + signature)
            diversity_threshold: 去重阈值

        Returns:
            list of (case, scores_dict) where scores_dict contains:
            - embedding_score: 向量相似度分数
            - signature_score: 签名匹配分数
            - venue_bonus: venue 匹配奖励
            - final_score: 最终融合分数
        """
        candidates = self.cases
        if venue_id:
            candidates = [c for c in candidates if c.venue_id == venue_id]

        if not candidates:
            return []

        # Step 1: Embedding retrieval
        embedding_results = self._embedding_retrieval(query_text, candidates)

        # Step 2: Signature retrieval (if signature provided and hybrid enabled)
        signature_results = {}
        if use_hybrid and signature:
            signature_results = self._signature_retrieval(signature, candidates)

        # Step 3: Merge and rerank
        merged = self._merge_and_rerank(
            embedding_results,
            signature_results,
            venue_id,
        )

        # Step 4: Diversity control
        merged = self._apply_diversity(merged, diversity_threshold)

        # Return top-k with scores
        return merged[:top_k]

    def _embedding_retrieval(
        self,
        query_text: str,
        candidates: list[PaperCase],
    ) -> dict[str, float]:
        """基于 embedding 的检索"""
        if not self.embedding_client:
            return {}

        # Get embeddings for cases without them
        cases_need_embedding = [c for c in candidates if not c.embedding]
        if cases_need_embedding:
            texts = [self.get_case_text(c) for c in cases_need_embedding]
            embeddings = self.embedding_client.embed(texts)
            for case, emb in zip(cases_need_embedding, embeddings):
                case.embedding = emb.tolist()
            self._save()

        query_vec = self.embedding_client.embed([query_text])[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)

        results: dict[str, float] = {}
        for case in candidates:
            if case.embedding:
                case_vec = np.array(case.embedding)
                case_vec = case_vec / (np.linalg.norm(case_vec) + 1e-12)
                score = float(np.dot(query_vec, case_vec))
                results[case.case_id] = score

        return results

    def _signature_retrieval(
        self,
        signature: PaperSignature,
        candidates: list[PaperCase],
    ) -> dict[str, float]:
        """基于 signature 的检索"""
        results: dict[str, float] = {}
        for case in candidates:
            score = self._signature_similarity(signature, case.paper_signature)
            results[case.case_id] = score
        return results

    def _merge_and_rerank(
        self,
        embedding_results: dict[str, float],
        signature_results: dict[str, float],
        target_venue_id: str | None,
    ) -> list[tuple[PaperCase, dict[str, float]]]:
        """融合并重排序"""
        all_case_ids = set(embedding_results.keys()) | set(signature_results.keys())

        scored: list[tuple[PaperCase, dict[str, float]]] = []
        for case_id in all_case_ids:
            case = self._index.get(case_id)
            if not case:
                continue

            emb_score = embedding_results.get(case_id, 0.0)
            sig_score = signature_results.get(case_id, 0.0)
            venue_bonus = self.venue_match_bonus if case.venue_id == target_venue_id else 0.0

            # Weighted fusion
            final_score = (
                self.embedding_weight * emb_score +
                self.signature_weight * sig_score +
                venue_bonus
            )

            scored.append((case, {
                "embedding_score": emb_score,
                "signature_score": sig_score,
                "venue_bonus": venue_bonus,
                "final_score": final_score,
            }))

        scored.sort(key=lambda x: x[1]["final_score"], reverse=True)
        return scored

    def _apply_diversity(
        self,
        results: list[tuple[PaperCase, dict[str, float]]],
        threshold: float,
    ) -> list[tuple[PaperCase, dict[str, float]]]:
        """
        多样性控制：
        - 相同 paper_id 只保留最高分
        - 避免全部来自同一 venue/year
        """
        seen_papers: set[str] = set()
        venue_counts: dict[str, int] = {}
        year_counts: dict[int, int] = {}

        diverse: list[tuple[PaperCase, dict[str, float]]] = []
        max_per_venue = max(3, len(results) // 3)  # 每个 venue 最多占 1/3
        max_per_year = max(3, len(results) // 3)

        for case, scores in results:
            # Skip duplicate papers
            if case.paper_id and case.paper_id in seen_papers:
                continue

            # Limit per venue
            venue = case.venue_id or "unknown"
            if venue_counts.get(venue, 0) >= max_per_venue:
                continue

            # Limit per year
            if case.year and year_counts.get(case.year, 0) >= max_per_year:
                continue

            # Accept this case
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
        """基于向量相似度搜索案例（向后兼容）"""
        results = self.retrieve_cases(
            query_text=query_text,
            signature=None,
            top_k=top_k,
            venue_id=venue_id,
            use_hybrid=False,
        )
        return [case for case, scores in results if scores["final_score"] >= threshold]

    def search_by_signature(
        self,
        signature: PaperSignature,
        top_k: int = 10,
        venue_id: str | None = None,
    ) -> list[PaperCase]:
        """基于签名特征搜索相似案例（向后兼容）"""
        candidates = self.list_cases(venue_id=venue_id)
        if not candidates:
            return []

        scored: list[tuple[PaperCase, float]] = []
        for case in candidates:
            score = self._signature_similarity(signature, case.paper_signature)
            scored.append((case, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [case for case, _ in scored[:top_k]]

    def _signature_similarity(self, sig1: PaperSignature, sig2: PaperSignature | None) -> float:
        """计算两个签名之间的相似度"""
        if sig2 is None:
            return 0.0
        score = 0.0
        # Paper type match
        if sig1.paper_type and sig2.paper_type and sig1.paper_type == sig2.paper_type:
            score += 0.2
        # Domain match
        if sig1.domain and sig2.domain and sig1.domain == sig2.domain:
            score += 0.2
        # Task overlap
        if sig1.tasks and sig2.tasks:
            overlap = len(set(sig1.tasks) & set(sig2.tasks))
            total = len(set(sig1.tasks) | set(sig2.tasks))
            score += 0.2 * (overlap / max(total, 1))
        # Method family overlap
        if sig1.method_family and sig2.method_family:
            overlap = len(set(sig1.method_family) & set(sig2.method_family))
            total = len(set(sig1.method_family) | set(sig2.method_family))
            score += 0.2 * (overlap / max(total, 1))
        # Dataset overlap
        if sig1.datasets and sig2.datasets:
            overlap = len(set(sig1.datasets) & set(sig2.datasets))
            total = len(set(sig1.datasets) | set(sig2.datasets))
            score += 0.2 * (overlap / max(total, 1))
        return score

    def update_case(self, case_id: str, updates: dict[str, Any]) -> None:
        """更新案例"""
        case = self._index.get(case_id)
        if case:
            for key, value in updates.items():
                if hasattr(case, key):
                    setattr(case, key, value)
            self._save()

    def delete_case(self, case_id: str) -> bool:
        """删除案例"""
        if case_id in self._index:
            case = self._index.pop(case_id)
            self.cases.remove(case)
            self._save()
            return True
        return False