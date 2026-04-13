"""DeepReviewCaseStore: DeepReview-13K 数据集的记忆存储

支持按 primary_area 过滤检索，作为热插拔记忆源
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
from storage.case_store import CaseStore, CASES_COLLECTION

logger = logging.getLogger(__name__)

DEEPREVIEW_COLLECTION = "deepreview_cases_v2"  # 使用新collection名避免schema冲突


class DeepReviewCaseStore(CaseStore):
    """DeepReview-13K 数据集的记忆存储，继承 CaseStore 并扩展 primary_area 支持

    特点：
    - 支持 primary_area 过滤检索（按研究方向过滤）
    - 支持从 DeepReview-13K 原始 JSONL 构建 PaperCase
    - 可作为热插拔记忆源与主 CaseStore 并行使用
    """

    def __init__(
        self,
        path: str | Path,
        embedding_client: EmbeddingClient | None = None,
        milvus_config: MilvusConfig | None = None,
        embedding_weight: float = 0.5,
        signature_weight: float = 0.4,
        venue_match_bonus: float = 0.1,
        primary_area_weight: float = 0.1,  # primary_area 匹配权重
    ) -> None:
        # 先初始化 _area_index，避免父类 _load 访问时出错
        self._area_index: dict[str, list[PaperCase]] = {}
        self.primary_area_weight = primary_area_weight
        super().__init__(
            path=path,
            embedding_client=embedding_client,
            milvus_config=milvus_config,
            embedding_weight=embedding_weight,
            signature_weight=signature_weight,
            venue_match_bonus=venue_match_bonus,
        )

    def _load(self) -> None:
        """加载时构建 primary_area 索引"""
        super()._load()
        # 构建 primary_area 索引
        for case in self.cases:
            if case.primary_area:
                if case.primary_area not in self._area_index:
                    self._area_index[case.primary_area] = []
                self._area_index[case.primary_area].append(case)
        logger.info(
            "Loaded %d DeepReview cases with %d unique areas",
            len(self.cases),
            len(self._area_index),
        )

    def get_cases_by_area(self, primary_area: str) -> list[PaperCase]:
        """根据 primary_area 获取案例"""
        return self._area_index.get(primary_area, [])

    def list_areas(self) -> list[str]:
        """列出所有 primary_area"""
        return list(self._area_index.keys())

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
        primary_area: str | None = None,  # 新增：按 primary_area 过滤
    ) -> list[tuple[PaperCase, dict[str, float]]]:
        """扩展的 retrieval，支持 primary_area 过滤

        Args:
            primary_area: 论文研究方向，用于过滤相似领域的案例
        """
        candidates = [
            c for c in self.cases
            if (not venue_id or c.venue_id == venue_id)
            and (not primary_area or c.primary_area == primary_area)
            and (not exclude_paper_id or c.paper_id != exclude_paper_id)
            and (not before_year or (c.year is not None and c.year < before_year))
        ]

        if not candidates:
            return []

        embedding_results = self._embedding_retrieval(query_text, candidates)

        signature_results = {}
        if use_hybrid and signature:
            signature_results = self._signature_retrieval(signature, candidates)

        merged = self._merge_and_rerank_with_area(
            embedding_results,
            signature_results,
            venue_id,
            primary_area,
        )

        merged = self._apply_diversity(merged, diversity_threshold)

        return merged[:top_k]

    def _merge_and_rerank_with_area(
        self,
        embedding_results: dict[str, float],
        signature_results: dict[str, float],
        target_venue_id: str | None,
        target_primary_area: str | None,
    ) -> list[tuple[PaperCase, dict[str, float]]]:
        """融合并重排序，包含 primary_area bonus"""
        all_case_ids = set(embedding_results.keys()) | set(signature_results.keys())

        scored: list[tuple[PaperCase, dict[str, float]]] = []
        for case_id in all_case_ids:
            case = self._index.get(case_id)
            if not case:
                continue

            emb_score = embedding_results.get(case_id, 0.0)
            sig_score = signature_results.get(case_id, 0.0)
            venue_bonus = self.venue_match_bonus if case.venue_id == target_venue_id else 0.0
            area_bonus = (
                self.primary_area_weight
                if case.primary_area == target_primary_area and target_primary_area
                else 0.0
            )

            final_score = (
                self.embedding_weight * emb_score +
                self.signature_weight * sig_score +
                venue_bonus +
                area_bonus
            )

            scored.append((case, {
                "embedding_score": emb_score,
                "signature_score": sig_score,
                "venue_bonus": venue_bonus,
                "area_bonus": area_bonus,
                "final_score": final_score,
            }))

        scored.sort(key=lambda x: x[1]["final_score"], reverse=True)
        return scored

    @classmethod
    def from_deepreview_jsonl(
        cls,
        jsonl_path: str | Path,
        output_path: str | Path,
        embedding_client: EmbeddingClient | None = None,
        milvus_config: MilvusConfig | None = None,
        venue_id: str = "DeepReview",
        limit: int | None = None,
    ) -> "DeepReviewCaseStore":
        """从 DeepReview-13K JSONL 构建 PaperCase 存储

        Args:
            jsonl_path: DeepReview-13K 数据集路径
            output_path: 输出的 PaperCase JSONL 路径
            embedding_client: 用于生成 embedding
            milvus_config: Milvus 配置
            venue_id: venue 标识
            limit: 限制处理数量（用于测试）
        """
        import re

        jsonl_path = Path(jsonl_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cases: list[PaperCase] = []
        count = 0

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if limit and count >= limit:
                    break

                data = json.loads(line)
                paper_text = data.get("paper", "")

                # 提取 title
                title_match = re.search(r"\\title\{([^}]+)\}", paper_text)
                title = title_match.group(1) if title_match else "Unknown Title"

                # 提取 abstract
                abstract_match = re.search(
                    r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                    paper_text,
                    re.DOTALL,
                )
                abstract = abstract_match.group(1) if abstract_match else ""
                # 清理 LaTeX
                abstract = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", abstract)
                abstract = re.sub(r"\\[a-zA-Z]+", "", abstract)
                abstract = abstract.strip()

                # 提取 strengths/weaknesses
                strengths: list[str] = []
                weaknesses: list[str] = []

                for rc in data.get("reviewer_comments", []):
                    content = rc.get("content", {})
                    if content.get("strengths"):
                        strengths.append(content["strengths"])
                    if content.get("weaknesses"):
                        weaknesses.append(content["weaknesses"])

                # 解析 rating（如 "[6, 6, 8, 8, 6]"）
                rating_raw = data.get("rating", "")
                rating_avg: float | None = None
                try:
                    # 尝试解析列表格式
                    if rating_raw.startswith("["):
                        ratings = json.loads(rating_raw)
                        rating_avg = sum(ratings) / len(ratings) if ratings else None
                    else:
                        # 尝试解析单个数字
                        rating_avg = float(rating_raw)
                except (json.JSONDecodeError, ValueError):
                    pass

                case = PaperCase(
                    case_id=f"deepreview_{data['id']}",
                    paper_id=data["id"],
                    venue_id=venue_id,
                    year=data.get("year"),
                    title=title,
                    abstract=abstract,
                    primary_area=data.get("primary_area"),
                    top_strengths=strengths[:3],  # 只取前3个
                    top_weaknesses=weaknesses[:3],
                    decision=data.get("decision"),
                    rating=rating_avg,
                )
                cases.append(case)
                count += 1

                if count % 100 == 0:
                    logger.info("Processed %d cases...", count)

        # 保存到文件
        write_json(output_path, [c.model_dump(exclude={"embedding"}) for c in cases])
        logger.info("Saved %d cases to %s", len(cases), output_path)

        # 创建 store 实例
        store = cls(
            path=output_path,
            embedding_client=embedding_client,
            milvus_config=milvus_config,
        )

        # 如果需要生成 embedding，批量处理（分批避免内存问题）
        if embedding_client:
            batch_size = 50

            for i in range(0, len(cases), batch_size):
                batch_cases = cases[i:i+batch_size]
                texts = [cls.get_case_text(c) for c in batch_cases]
                embeddings = embedding_client.embed(texts)

                for case, emb in zip(batch_cases, embeddings):
                    case.embedding = emb.tolist()

                if (i + batch_size) % 500 == 0 or i + batch_size >= len(cases):
                    logger.info("Generated embeddings for %d/%d cases...", min(i+batch_size, len(cases)), len(cases))

            # 存到 Milvus（只存 embedding，不存 text）
            if milvus_config:
                milvus = MilvusStore(milvus_config)
                try:
                    # 分批插入（只存 ID 和 embedding，不存 text）
                    for i in range(0, len(cases), batch_size):
                        batch_cases = cases[i:i+batch_size]
                        milvus.upsert_embeddings(
                            DEEPREVIEW_COLLECTION,
                            [c.case_id for c in batch_cases],
                            [c.embedding for c in batch_cases if c.embedding],
                            texts=None,  # 不存 text，元数据在 JSONL 中
                        )

                    logger.info("Stored %d embeddings in Milvus collection %s", len(cases), DEEPREVIEW_COLLECTION)
                except Exception as e:
                    logger.warning("Failed to store embeddings in Milvus: %s", e)

        return store