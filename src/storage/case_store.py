"""CaseStore: 论文案例存储和检索，支持 hybrid retrieval

优化版本：
- Metadata 存在 JSON（快速加载）
- Embedding 存在 Milvus（向量检索）
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from clients.embedding_client import EmbeddingClient
from common.types import PaperCase, PaperSignature
from common.utils import write_json
from storage.milvus_store import MilvusConfig, MilvusStore

logger = logging.getLogger(__name__)

# 默认 Milvus collection 名称
CASES_COLLECTION = "paper_cases"


class CaseStore:
    """管理 PaperCase 的存储和检索，支持 embedding + signature hybrid retrieval"""

    def __init__(
        self,
        path: str | Path,
        embedding_client: EmbeddingClient | None = None,
        milvus_config: MilvusConfig | None = None,
        embedding_weight: float = 0.5,
        signature_weight: float = 0.4,
        venue_match_bonus: float = 0.1,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_client = embedding_client
        self.milvus_config = milvus_config
        self.embedding_weight = embedding_weight
        self.signature_weight = signature_weight
        self.venue_match_bonus = venue_match_bonus
        self.cases: list[PaperCase] = []
        self._index: dict[str, PaperCase] = {}
        self._milvus: MilvusStore | None = None

        if self.path.exists():
            self._load()

    def _get_milvus(self) -> MilvusStore | None:
        """获取 Milvus 连接（懒加载）"""
        if self._milvus is None and self.milvus_config:
            try:
                self._milvus = MilvusStore(self.milvus_config)
            except Exception as e:
                logger.warning("Failed to connect to Milvus: %s", e)
        return self._milvus

    def _load(self) -> None:
        """加载 metadata（不含 embedding），embedding 从 Milvus 按需获取"""
        with open(self.path, encoding="utf-8") as f:
            content = f.read().strip()

        # 支持 JSON 数组或 JSONL 格式
        if content.startswith('['):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split('\n') if line.strip()]

        # 不加载 embedding 到内存
        for item in data:
            if "embedding" in item:
                del item["embedding"]
            case = PaperCase(**item)
            self.cases.append(case)
            self._index[case.case_id] = case

        logger.info("Loaded %d cases from %s", len(self.cases), self.path)

    def _save(self) -> None:
        """保存 metadata（不含 embedding）"""
        data = []
        for case in self.cases:
            d = case.model_dump()
            # 不保存 embedding 到 JSON
            if "embedding" in d:
                del d["embedding"]
            data.append(d)
        write_json(self.path, data)

    def _get_case_embedding(self, case: PaperCase) -> np.ndarray | None:
        """获取 case 的 embedding（从 Milvus 或实时生成）"""
        if case.embedding:
            return np.array(case.embedding)

        # 尝试从 Milvus 获取
        milvus = self._get_milvus()
        if milvus:
            try:
                texts = milvus.search_texts(CASES_COLLECTION, [], 1)  # placeholder
                # 实际应该用 doc_id 查询
            except:
                pass

        # 实时生成
        if self.embedding_client:
            text = self.get_case_text(case)
            emb = self.embedding_client.embed([text])[0]
            case.embedding = emb.tolist()
            return emb

        return None

    @staticmethod
    def get_case_text(case: PaperCase) -> str:
        """获取案例的文本表示，用于 embedding"""
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

        # 生成 embedding 并存到 Milvus
        if self.embedding_client and not case.embedding:
            text = self.get_case_text(case)
            emb = self.embedding_client.embed([text])[0]
            case.embedding = emb.tolist()

            # 存到 Milvus
            milvus = self._get_milvus()
            if milvus:
                try:
                    milvus.upsert_embeddings(
                        CASES_COLLECTION,
                        [case.case_id],
                        [case.embedding],
                        texts=[text],
                    )
                except Exception as e:
                    logger.warning("Failed to store embedding in Milvus: %s", e)

        # 只存 metadata 到内存和 JSON
        case_copy = case.model_copy()
        case_copy.embedding = None  # 内存里不存 embedding
        self._index[case.case_id] = case_copy
        self.cases.append(case_copy)
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
        exclude_paper_id: str | None = None,
        before_year: int | None = None,
    ) -> list[tuple[PaperCase, dict[str, float]]]:
        """统一 hybrid retrieval 入口

        Args:
            query_text: 查询文本
            signature: 论文签名
            top_k: 返回数量
            venue_id: 过滤 venue
            use_hybrid: 是否使用混合检索
            diversity_threshold: 多样性阈值
            exclude_paper_id: 排除的 paper_id（避免数据泄露）
            before_year: 只返回该年份之前的案例
        """
        candidates = self.cases
        if venue_id:
            candidates = [c for c in candidates if c.venue_id == venue_id]

        # 排除目标论文（避免数据泄露）
        if exclude_paper_id:
            candidates = [c for c in candidates if c.paper_id != exclude_paper_id]

        # 只使用目标年份之前的案例
        if before_year:
            candidates = [c for c in candidates if c.year is not None and c.year < before_year]

        if not candidates:
            return []

        # Step 1: Embedding retrieval (使用 Milvus 或内存计算)
        embedding_results = self._embedding_retrieval(query_text, candidates)

        # Step 2: Signature retrieval
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

        return merged[:top_k]

    def _embedding_retrieval(
        self,
        query_text: str,
        candidates: list[PaperCase],
    ) -> dict[str, float]:
        """基于 embedding 的检索 - 优先使用 Milvus"""
        if not self.embedding_client:
            return {}

        query_vec = self.embedding_client.embed([query_text])[0].tolist()

        # 构建 candidates 的 case_id 集合，用于过滤
        candidate_ids = {c.case_id for c in candidates}

        # 尝试用 Milvus 检索
        milvus = self._get_milvus()
        if milvus and len(candidates) > 100:
            try:
                case_ids = milvus.search_ids(CASES_COLLECTION, query_vec, top_k=min(len(candidates) * 2, 100))
                # 只保留在 candidates 中的结果（过滤掉被排除的论文）
                results: dict[str, float] = {}
                for i, case_id in enumerate(case_ids):
                    if case_id in candidate_ids:
                        results[case_id] = 1.0 - (len(results) * 0.01)
                        if len(results) >= min(len(candidates), 50):
                            break
                logger.info("Milvus retrieved %d cases (filtered from %d)", len(results), len(case_ids))
                # 如果 Milvus 返回有效结果，直接返回
                if results:
                    return results
                # 否则继续使用暴力检索
                logger.info("Milvus results empty, falling back to brute force")
            except Exception as e:
                logger.warning("Milvus search failed, falling back to brute force: %s", e)

        # 回退到暴力搜索
        query_vec_np = np.array(query_vec)
        query_vec_np = query_vec_np / (np.linalg.norm(query_vec_np) + 1e-12)

        results: dict[str, float] = {}
        # 分批处理避免内存问题
        batch_size = 500
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            texts = [self.get_case_text(c) for c in batch]
            embeddings = self.embedding_client.embed(texts)

            for case, emb in zip(batch, embeddings):
                case_vec = emb / (np.linalg.norm(emb) + 1e-12)
                score = float(np.dot(query_vec_np, case_vec))
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
        if sig1.paper_type and sig2.paper_type and sig1.paper_type == sig2.paper_type:
            score += 0.2
        if sig1.domain and sig2.domain and sig1.domain == sig2.domain:
            score += 0.2
        if sig1.tasks and sig2.tasks:
            overlap = len(set(sig1.tasks) & set(sig2.tasks))
            total = len(set(sig1.tasks) | set(sig2.tasks))
            score += 0.2 * (overlap / max(total, 1))
        if sig1.method_family and sig2.method_family:
            overlap = len(set(sig1.method_family) & set(sig2.method_family))
            total = len(set(sig1.method_family) | set(sig2.method_family))
            score += 0.2 * (overlap / max(total, 1))
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