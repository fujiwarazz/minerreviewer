"""VectorMemoryStore: 支持向量检索的记忆存储

架构：
- metadata 存在 JSONL（快速加载、人类可读）
- embedding 存在 Milvus/FAISS（向量检索）
- 混合检索：embedding相似度 + metadata过滤 + utility权重
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from clients.embedding_client import EmbeddingClient
from common.types import ExperienceCard
from storage.milvus_store import MilvusConfig, MilvusStore
from storage.faiss_index import FaissIndex

logger = logging.getLogger(__name__)

# 默认 Milvus collection 名称
MEMORY_COLLECTION = "agent_memories"


class VectorMemoryStore:
    """管理 ExperienceCard 的存储和检索，支持向量检索"""

    def __init__(
        self,
        path: str | Path,
        embedding_client: EmbeddingClient | None = None,
        milvus_config: MilvusConfig | None = None,
        faiss_index_path: str | Path | None = None,
        embedding_weight: float = 0.6,
        metadata_weight: float = 0.3,
        utility_weight: float = 0.1,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_client = embedding_client
        self.milvus_config = milvus_config
        self.faiss_index_path = Path(faiss_index_path) if faiss_index_path else None

        # 检索权重
        self.embedding_weight = embedding_weight
        self.metadata_weight = metadata_weight
        self.utility_weight = utility_weight

        self.cards: list[ExperienceCard] = []
        self._index: dict[str, ExperienceCard] = {}
        self._milvus: MilvusStore | None = None
        self._faiss: FaissIndex | None = None

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
        """加载 metadata（不含 embedding）"""
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
            card = ExperienceCard(**item)
            self.cards.append(card)
            self._index[card.card_id] = card

        logger.info("Loaded %d cards from %s", len(self.cards), self.path)

    def _save(self) -> None:
        """保存 metadata（不含 embedding）"""
        data = []
        for card in self.cards:
            d = card.model_dump(mode='json')  # 使用 mode='json' 正确序列化 datetime
            # 不保存 embedding 到 JSON（存到Milvus/FAISS）
            if "embedding" in d:
                del d["embedding"]
            data.append(d)

        # JSONL 格式保存
        with open(self.path, 'w', encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def add_card(self, card: ExperienceCard, owner_agent: str | None = None) -> str:
        """添加卡片，返回 card_id

        Args:
            card: 要添加的经验卡片
            owner_agent: 所属agent（可选，会覆盖card.owner_agent）

        Returns:
            card_id
        """
        if not card.card_id:
            card.card_id = str(uuid.uuid4())

        if owner_agent:
            card.owner_agent = owner_agent

        # 生成 embedding 并存到 Milvus/FAISS
        if self.embedding_client:
            text = ExperienceCard.get_card_text(card)
            emb = self.embedding_client.embed([text])[0]
            card.embedding = emb.tolist()

            # 存到 Milvus
            milvus = self._get_milvus()
            if milvus:
                try:
                    milvus.upsert_embeddings(
                        MEMORY_COLLECTION,
                        [card.card_id],
                        [card.embedding],
                        texts=[text],
                    )
                except Exception as e:
                    logger.warning("Failed to store embedding in Milvus: %s", e)

            # 或存到 FAISS
            elif self.faiss_index_path:
                self._update_faiss_index()

        # 只存 metadata 到内存和 JSON
        card_copy = card.model_copy()
        card_copy.embedding = None  # 内存里不存 embedding
        self._index[card.card_id] = card_copy
        self.cards.append(card_copy)
        self._save()
        logger.info("Added card %s (owner=%s, kind=%s)", card.card_id[:8], card.owner_agent, card.kind)
        return card.card_id

    def batch_add_cards(
        self,
        cards: list[ExperienceCard],
        owner_agent: str | None = None
    ) -> list[str]:
        """批量添加卡片

        Args:
            cards: 卡片列表
            owner_agent: 所属agent

        Returns:
            list of card_ids
        """
        card_ids = []

        # 批量生成embedding
        if self.embedding_client and cards:
            texts = [ExperienceCard.get_card_text(c) for c in cards]
            embeddings = self.embedding_client.embed(texts)

            for card, emb in zip(cards, embeddings):
                if not card.card_id:
                    card.card_id = str(uuid.uuid4())
                if owner_agent:
                    card.owner_agent = owner_agent
                card.embedding = emb.tolist()
                card_ids.append(card.card_id)

            # 批量存到Milvus
            milvus = self._get_milvus()
            if milvus:
                try:
                    milvus.upsert_embeddings(
                        MEMORY_COLLECTION,
                        card_ids,
                        [c.embedding for c in cards],
                        texts=texts,
                    )
                except Exception as e:
                    logger.warning("Failed to batch store embeddings in Milvus: %s", e)

            # 存metadata
            for card in cards:
                card_copy = card.model_copy()
                card_copy.embedding = None
                self._index[card.card_id] = card_copy
                self.cards.append(card_copy)

            self._save()
            logger.info("Batch added %d cards (owner=%s)", len(cards), owner_agent)

        return card_ids

    def retrieve_cards(
        self,
        query_text: str,
        owner_agent: str | None = None,
        kind: Literal["strength", "critique", "failure"] | None = None,
        theme: str | None = None,
        top_k: int = 10,
        min_utility: float = 0.0,
        use_vector_search: bool = True,
        primary_area: str | None = None,  # 新增：按领域过滤
    ) -> list[tuple[ExperienceCard, dict[str, float]]]:
        """向量检索卡片

        Args:
            query_text: 查询文本
            owner_agent: 过滤特定agent的记忆
            kind: 过滤卡片类型
            theme: 过滤主题
            top_k: 返回数量
            min_utility: 最小utility阈值
            use_vector_search: 是否使用向量检索
            primary_area: 目标领域，支持三级匹配策略

        Returns:
            list of (card, scores_dict)
        """
        # Step 1: Metadata 过滤（含领域三级匹配）
        candidates = self.cards

        if owner_agent:
            candidates = [c for c in candidates if c.owner_agent == owner_agent]
        if kind:
            candidates = [c for c in candidates if c.kind == kind]
        if theme:
            candidates = [c for c in candidates if c.theme == theme]
        if min_utility > 0:
            candidates = [c for c in candidates if c.utility >= min_utility]

        # 领域三级匹配策略
        if primary_area:
            level1 = []  # 精确匹配：同领域
            level2 = []  # 泛化卡片：primary_area=None
            level3 = []  # 跨领域：不同领域

            for c in candidates:
                if c.primary_area == primary_area:
                    level1.append(c)
                elif c.primary_area is None:
                    level2.append(c)
                else:
                    level3.append(c)

            # 优先填充 level1，不足时用 level2，再不足用 level3
            candidates = level1 + level2 + level3
        else:
            # 无领域指定时，通用卡片优先
            generic = [c for c in candidates if c.primary_area is None]
            specific = [c for c in candidates if c.primary_area is not None]
            candidates = generic + specific

        if not candidates:
            return []

        # Step 2: 向量检索（可选）
        embedding_scores: dict[str, float] = {}
        if use_vector_search and self.embedding_client:
            embedding_scores = self._embedding_retrieval(query_text, candidates)

        # Step 3: 综合评分和排序
        # 领域降权因子
        area_penalties: dict[str, float] = {}
        for c in candidates:
            if primary_area:
                if c.primary_area == primary_area:
                    area_penalties[c.card_id] = 1.0  # 精确匹配
                elif c.primary_area is None:
                    area_penalties[c.card_id] = 0.9  # 通用卡片
                else:
                    area_penalties[c.card_id] = 0.7  # 跨领域
            else:
                if c.primary_area is None:
                    area_penalties[c.card_id] = 1.0  # 通用卡片优先
                else:
                    area_penalties[c.card_id] = 0.8  # 有领域卡片降权

        scored = self._score_and_rank(candidates, embedding_scores, area_penalties)

        return scored[:top_k]

    def _embedding_retrieval(
        self,
        query_text: str,
        candidates: list[ExperienceCard],
    ) -> dict[str, float]:
        """基于 embedding 的检索 - 优先使用 Milvus"""
        if not self.embedding_client:
            return {}

        query_vec = self.embedding_client.embed([query_text])[0].tolist()
        candidate_ids = {c.card_id for c in candidates}

        # 尝试用 Milvus 检索
        milvus = self._get_milvus()
        if milvus and len(candidates) > 50:
            try:
                card_ids = milvus.search_ids(
                    MEMORY_COLLECTION,
                    query_vec,
                    top_k=min(len(candidates) * 2, 100)
                )
                results: dict[str, float] = {}
                for i, card_id in enumerate(card_ids):
                    if card_id in candidate_ids:
                        # 简化的分数计算（Milvus返回的顺序代表相似度）
                        results[card_id] = 1.0 - (len(results) * 0.01)
                        if len(results) >= min(len(candidates), 50):
                            break
                logger.info("Milvus retrieved %d cards", len(results))
                if results:
                    return results
            except Exception as e:
                logger.warning("Milvus search failed: %s", e)

        # 回退到暴力搜索
        query_vec_np = np.array(query_vec)
        query_vec_np = query_vec_np / (np.linalg.norm(query_vec_np) + 1e-12)

        results: dict[str, float] = {}
        batch_size = 500
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            texts = [ExperienceCard.get_card_text(c) for c in batch]
            embeddings = self.embedding_client.embed(texts)

            for card, emb in zip(batch, embeddings):
                card_vec = emb / (np.linalg.norm(emb) + 1e-12)
                score = float(np.dot(query_vec_np, card_vec))
                results[card.card_id] = score

        return results

    def _score_and_rank(
        self,
        candidates: list[ExperienceCard],
        embedding_scores: dict[str, float],
        area_penalties: dict[str, float] | None = None,
    ) -> list[tuple[ExperienceCard, dict[str, float]]]:
        """综合评分和排序

        Args:
            candidates: 候选卡片列表
            embedding_scores: 向量相似度分数
            area_penalties: 领域降权因子（card_id → factor）
        """
        area_penalties = area_penalties or {}
        scored: list[tuple[ExperienceCard, dict[str, float]]] = []

        for card in candidates:
            emb_score = embedding_scores.get(card.card_id, 0.0)

            # Metadata分数：基于utility和confidence
            meta_score = (card.utility or 0.5) * 0.6 + (card.confidence or 0.5) * 0.4

            # 综合分数
            final_score = (
                self.embedding_weight * emb_score +
                self.metadata_weight * meta_score +
                self.utility_weight * (card.utility or 0.5)
            )

            # 领域降权
            area_penalty = area_penalties.get(card.card_id, 1.0)
            if area_penalty < 1.0:
                final_score *= area_penalty

            scored.append((card, {
                "embedding_score": emb_score,
                "metadata_score": meta_score,
                "utility_score": card.utility or 0.5,
                "area_penalty": area_penalty,
                "final_score": final_score,
            }))

        scored.sort(key=lambda x: x[1]["final_score"], reverse=True)
        return scored

    def _update_faiss_index(self) -> None:
        """更新 FAISS 索引"""
        if not self.faiss_index_path or not self.embedding_client:
            return

        # 重新构建索引
        texts = [ExperienceCard.get_card_text(c) for c in self.cards]
        embeddings = self.embedding_client.embed(texts)
        ids = [c.card_id for c in self.cards]

        self._faiss = FaissIndex(
            self.faiss_index_path,
            self.faiss_index_path.with_suffix('.meta.json')
        )
        self._faiss.build(embeddings, ids)
        self._faiss.save()
        logger.info("Updated FAISS index with %d vectors", len(ids))

    def get_card(self, card_id: str) -> ExperienceCard | None:
        """根据 ID 获取卡片"""
        return self._index.get(card_id)

    def list_cards(
        self,
        owner_agent: str | None = None,
        kind: Literal["strength", "critique", "failure"] | None = None,
    ) -> list[ExperienceCard]:
        """列出卡片"""
        cards = self.cards
        if owner_agent:
            cards = [c for c in cards if c.owner_agent == owner_agent]
        if kind:
            cards = [c for c in cards if c.kind == kind]
        return cards

    def update_card(self, card_id: str, updates: dict[str, Any]) -> None:
        """更新卡片属性"""
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
        self._index = {c.card_id: c for c in self.cards}
        removed = original_count - len(self.cards)
        if removed > 0:
            self._save()
            logger.info("Removed %d inactive cards", removed)
        return removed

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_cards": len(self.cards),
            "active_cards": len([c for c in self.cards if c.active]),
            "by_kind": {},
            "by_owner": {},
        }

        for card in self.cards:
            kind = card.kind
            stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1

            owner = card.owner_agent or "unassigned"
            stats["by_owner"][owner] = stats["by_owner"].get(owner, 0) + 1

        return stats

    # === Effectiveness Tracking ===

    def record_usage(self, card_id: str, paper_id: str) -> None:
        """记录卡片被某篇论文的 review 使用"""
        card = self.get_card(card_id)
        if not card:
            return

        card.use_count += 1
        card.use_history.append({
            "paper_id": paper_id,
            "timestamp": str(datetime.utcnow()),
            "outcome": "pending",  # 等待对比学习结果
        })
        self._save()

    def apply_feedback(
        self,
        card_outcomes: dict[str, str],  # {card_id: "positive"|"negative"|"neutral"}
        feedback_paper_id: str,
    ) -> dict[str, Any]:
        """根据对比学习结果更新卡片质量

        Args:
            card_outcomes: {card_id: outcome}
                - "positive": pred和GT一致，卡片帮助了正确评价
                - "negative": pred和GT不一致，卡片可能误导
                - "neutral": 无法判断
            feedback_paper_id: 对应的论文 ID

        Returns:
            stats dict with updates/degraded/retired counts
        """
        stats: dict[str, Any] = {"updated": 0, "degraded": 0, "retired": 0, "promoted": 0}

        for card_id, outcome in card_outcomes.items():
            card = self.get_card(card_id)
            if not card:
                continue

            # 更新 use_history 中对应的 pending 记录
            for entry in card.use_history:
                if entry.get("outcome") == "pending" and entry.get("paper_id") == feedback_paper_id:
                    entry["outcome"] = outcome
                    break

            old_utility = card.utility

            if outcome == "positive":
                # 卡片帮助了正确评价 → 提升 utility 和 confidence
                card.utility = min(1.0, card.utility + 0.08)
                card.confidence = min(1.0, card.confidence + 0.05)
                stats["promoted"] += 1

            elif outcome == "negative":
                # 卡片可能误导 → 降低 utility
                card.utility = max(0.0, card.utility - 0.15)
                card.confidence = max(0.0, card.confidence - 0.1)
                stats["degraded"] += 1

                # 连续2次负面 → 退役
                recent = [e for e in card.use_history if e.get("paper_id") == feedback_paper_id or True]
                neg_count = sum(1 for e in card.use_history[-5:] if e.get("outcome") == "negative")
                if neg_count >= 2 and card.utility < 0.3:
                    card.active = False
                    stats["retired"] += 1
                    logger.info("Card %s retired (utility=%.2f, %d negative outcomes)", card_id[:8], card.utility, neg_count)

            else:  # neutral
                # 不改变
                pass

            if abs(card.utility - old_utility) > 0.01:
                stats["updated"] += 1

        if stats["updated"] > 0:
            self._save()

        if stats["retired"] > 0 or stats["degraded"] > 0:
            logger.info(
                "Feedback applied: %d updated, %d promoted, %d degraded, %d retired",
                stats["updated"], stats["promoted"], stats["degraded"], stats["retired"],
            )

        return stats

    def get_low_quality_cards(self, utility_threshold: float = 0.3) -> list[ExperienceCard]:
        """获取低质量卡片列表"""
        return [c for c in self.cards if c.utility < utility_threshold and c.active]

    def get_top_cards(self, top_n: int = 20, kind: str | None = None) -> list[ExperienceCard]:
        """获取高质量卡片列表"""
        cards = self.cards
        if kind:
            cards = [c for c in cards if c.kind == kind]
        cards.sort(key=lambda c: (c.utility or 0.5) + (c.confidence or 0.5), reverse=True)
        return cards[:top_n]