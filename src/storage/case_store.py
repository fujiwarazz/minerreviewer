"""CaseStore: 论文案例存储和检索"""
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
    """管理 PaperCase 的存储和检索"""

    def __init__(
        self,
        path: str | Path,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_client = embedding_client
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

    def add_case(self, case: PaperCase) -> str:
        """添加案例，返回 case_id"""
        if not case.case_id:
            case.case_id = str(uuid.uuid4())
        if self.embedding_client and not case.embedding:
            text = f"{case.title}\n{case.abstract}"
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

    def search_similar_cases(
        self,
        query_text: str,
        top_k: int = 10,
        venue_id: str | None = None,
        threshold: float = 0.0,
    ) -> list[PaperCase]:
        """基于向量相似度搜索案例"""
        if not self.embedding_client:
            logger.warning("No embedding client, returning empty results")
            return []
        if not self.cases:
            return []

        # Filter by venue first
        candidates = self.cases
        if venue_id:
            candidates = [c for c in candidates if c.venue_id == venue_id]

        # Get embeddings for cases without them
        cases_need_embedding = [c for c in candidates if not c.embedding]
        if cases_need_embedding:
            texts = [f"{c.title}\n{c.abstract}" for c in cases_need_embedding]
            embeddings = self.embedding_client.embed(texts)
            for case, emb in zip(cases_need_embedding, embeddings):
                case.embedding = emb.tolist()
            self._save()

        # Compute similarity
        query_vec = self.embedding_client.embed([query_text])[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)

        scored: list[tuple[PaperCase, float]] = []
        for case in candidates:
            if case.embedding:
                case_vec = np.array(case.embedding)
                case_vec = case_vec / (np.linalg.norm(case_vec) + 1e-12)
                score = float(np.dot(query_vec, case_vec))
                if score >= threshold:
                    scored.append((case, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [case for case, _ in scored[:top_k]]

    def search_by_signature(
        self,
        signature: PaperSignature,
        top_k: int = 10,
        venue_id: str | None = None,
    ) -> list[PaperCase]:
        """基于签名特征搜索相似案例"""
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