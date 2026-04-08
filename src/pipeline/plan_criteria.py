"""CriteriaPlanner: 规划和激活审稿标准"""
from __future__ import annotations

import logging
from collections import Counter

from clients.llm_client import LLMClient
from common.types import ActivatedCriterion, Criterion, ExperienceCard, PaperCase, PaperSignature, RetrievalBundle, VenuePolicy

logger = logging.getLogger(__name__)


class CriteriaPlanner:
    """规划和激活审稿标准"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def plan(
        self,
        signature: PaperSignature | None,
        bundle: RetrievalBundle,
        mined_criteria: list[Criterion],
        mined_policy_criteria: list[Criterion] | None = None,
        max_criteria: int = 15,
        priority_boost_memory: float = 1.5,
    ) -> list[ActivatedCriterion]:
        """
        规划审稿标准，返回激活的标准列表

        Args:
            signature: 论文结构化特征
            bundle: 多通道检索结果
            mined_criteria: 从历史数据挖掘的内容标准
            mined_policy_criteria: 从历史数据挖掘的策略标准
            max_criteria: 最大标准数量
            priority_boost_memory: 记忆来源的优先级加成

        Returns:
            激活的标准列表，按优先级排序
        """
        activated: list[ActivatedCriterion] = []
        mined_policy_criteria = mined_policy_criteria or []

        # 1. Add criteria from policy cards (memory) - venue-aware policy memory
        # Filter policy cards by relevance to paper signature
        # 限制 policy cards 数量，给其他 source 留出空间
        max_policy_cards = min(5, max_criteria // 3)  # 最多 5 个或 1/3
        relevant_policy_cards = self._filter_relevant_policy_cards(
            bundle.policy_cards, signature, max_cards=max_policy_cards
        )
        for card in relevant_policy_cards:
            priority = int((card.utility or 0.5) * 10)  # 移除 priority_boost_memory，降低优先级
            activated.append(ActivatedCriterion(
                theme=card.theme or "quality",
                criterion=card.content,
                source="policy_memory",  # 明确标记来源
                priority=priority,
                trigger_reason=f"Retrieved from venue policy memory (venue={card.venue_id}, utility={card.utility:.2f})",
                required_evidence=card.trigger,
                owner_agent=f"theme_{card.theme or 'quality'}",
            ))
            # Update use count
            card.use_count += 1

        # 2. Add criteria from similar paper cases (case-driven evidence)
        for case in bundle.similar_paper_cases:
            for criterion_text in case.transferable_criteria:
                theme = self._infer_theme(criterion_text)
                priority = 8  # 提高优先级，与 policy_mined 竞争
                if case.decision == "reject":
                    priority += 2  # Boost for rejected papers
                activated.append(ActivatedCriterion(
                    theme=theme,
                    criterion=criterion_text,
                    source="case_memory",
                    priority=priority,
                    trigger_reason=f"Transferred from similar case {case.case_id[:8]} (decision={case.decision})",
                    owner_agent=f"theme_{theme}",
                ))

        # 3. Add criteria from failure cards
        for card in bundle.failure_cards:
            theme = card.theme or "quality"
            activated.append(ActivatedCriterion(
                theme=theme,
                criterion=card.content,
                source="failure_memory",
                priority=9,  # High priority for failure patterns，与 policy_mined 相同
                trigger_reason=f"Failure pattern: {card.trigger}",
                required_evidence=card.trigger,
                owner_agent=f"theme_{theme}",
            ))

        # 4. Add mined policy criteria (mined from accept/reject reviews) - HIGH PRIORITY
        for criterion in mined_policy_criteria:
            priority = 9  # Higher priority than policy cards - specific to this paper
            activated.append(ActivatedCriterion(
                theme=criterion.theme,
                criterion=criterion.text,
                source="policy_mined",
                priority=priority,
                trigger_reason=f"Mined from {criterion.kind} criteria (venue policy)",
                owner_agent=f"theme_{criterion.theme}",
            ))

        # 5. Add mined content criteria - HIGH PRIORITY
        for criterion in mined_criteria:
            priority = 8  # High priority - specific to this paper's content
            activated.append(ActivatedCriterion(
                theme=criterion.theme,
                criterion=criterion.text,
                source="content_mined",
                priority=priority,
                trigger_reason=f"Mined from {criterion.kind} criteria",
                owner_agent=f"theme_{criterion.theme}",
            ))

        # 6. Balance themes with LLM if severely imbalanced
        theme_counts = Counter(c.theme for c in activated)
        max_theme_ratio = max(theme_counts.values()) / len(activated) if activated else 0

        if max_theme_ratio > 0.7 and signature:  # 如果某主题占比超过70%
            logger.info("Theme imbalance detected (%.1f%%), using LLM to rebalance...", max_theme_ratio * 100)
            activated = self._rebalance_priorities_with_llm(activated, signature)

        # 7. Deduplicate and merge similar criteria
        activated = self._deduplicate(activated)

        # 8. Sort by priority and limit
        activated.sort(key=lambda x: x.priority, reverse=True)
        activated = activated[:max_criteria]

        # Log source distribution
        source_counts = Counter(c.source for c in activated)
        logger.info(
            "Planned %d activated criteria: %s",
            len(activated),
            dict(source_counts)
        )
        return activated

    def _infer_theme(self, criterion_text: str) -> str:
        """根据标准文本推断主题"""
        text_lower = criterion_text.lower()
        theme_keywords = {
            "novelty": ["novel", "new", "original", "contribution", "innovation"],
            "quality": ["quality", "correctness", "accuracy", "performance", "baseline"],
            "clarity": ["clarity", "writing", "presentation", "readable", "explain"],
            "significance": ["significant", "impact", "important", "influence", "application"],
            "reproducibility": ["reproducib", "code", "implementation", "detail", "experiment"],
            "soundness": ["sound", "theory", "proof", "mathematical", "formal"],
            "methodology": ["method", "approach", "technique", "design", "architecture"],
        }
        for theme, keywords in theme_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return theme
        return "quality"

    def _filter_relevant_policy_cards(
        self,
        policy_cards: list[ExperienceCard],
        signature: PaperSignature | None,
        max_cards: int = 20,
    ) -> list[ExperienceCard]:
        """根据论文特征筛选相关的 policy cards

        策略：
        1. 所有论文都需要通用主题：quality, clarity
        2. 根据 paper_type 动态调整相关主题
        3. 始终包含高 utility 的 cards，确保覆盖度
        """
        if not policy_cards:
            return []

        # 基础主题（所有论文都需要）
        relevant_themes = {"quality", "clarity"}

        if signature:
            paper_type = (signature.paper_type or "").lower()

            # 根据论文类型调整主题
            if "theoretical" in paper_type:
                relevant_themes.update({"soundness", "significance", "novelty"})
            elif "empirical" in paper_type or "experimental" in paper_type:
                relevant_themes.update({"reproducibility", "methodology"})
            elif "survey" in paper_type or "review" in paper_type:
                relevant_themes.update({"significance", "clarity", "completeness"})
            else:
                # 默认：包含所有主题
                relevant_themes.update({"soundness", "significance", "novelty", "reproducibility"})

        # 筛选相关主题的 cards
        relevant_cards = [
            card for card in policy_cards
            if (card.theme or "quality") in relevant_themes
        ]

        # 按 utility + confidence 排序
        relevant_cards.sort(
            key=lambda c: (c.utility or 0.5) + (c.confidence or 0.5),
            reverse=True
        )

        return relevant_cards[:max_cards]

    def _deduplicate(self, criteria: list[ActivatedCriterion]) -> list[ActivatedCriterion]:
        """去重相似的标准"""
        if not criteria:
            return criteria

        # Group by theme
        by_theme: dict[str, list[ActivatedCriterion]] = {}
        for c in criteria:
            if c.theme not in by_theme:
                by_theme[c.theme] = []
            by_theme[c.theme].append(c)

        # Deduplicate within each theme
        result: list[ActivatedCriterion] = []
        for theme, items in by_theme.items():
            seen_texts: set[str] = set()
            for item in items:
                # Simple text deduplication
                text_key = item.criterion.lower()[:50]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    result.append(item)
                else:
                    # Merge priorities if duplicate
                    for existing in result:
                        if existing.criterion.lower()[:50] == text_key:
                            existing.priority = max(existing.priority, item.priority)
                            break

        return result

    def _rebalance_priorities_with_llm(
        self,
        activated: list[ActivatedCriterion],
        signature: PaperSignature,
    ) -> list[ActivatedCriterion]:
        """用 LLM 批量重新评估 priority，解决主题不平衡（1次调用）"""
        if len(activated) < 5:
            return activated

        # 构建 criteria 摘要
        criteria_summary = []
        for i, ac in enumerate(activated):
            summary = f"{i}. [{ac.theme}] {ac.criterion[:80]}..."
            criteria_summary.append(summary)

        prompt = f"""As an ICLR reviewer coordinator, evaluate the relevance of these review criteria for a paper.

Paper Info:
- Type: {signature.paper_type or 'research'}
- Domain: {signature.domain or 'general'}
- Tasks: {', '.join(signature.tasks[:3]) if signature.tasks else 'N/A'}
- Method: {', '.join(signature.method_family[:3]) if signature.method_family else 'N/A'}

Criteria to evaluate:
{chr(10).join(criteria_summary)}

For EACH criterion, rate:
1. relevance (1-10): How relevant to THIS specific paper?
2. theme_appropriate (1-10): Is the theme appropriate for this paper type?

Return JSON:
{{
  "evaluations": [
    {{"index": 0, "relevance": 8, "theme_appropriate": 9, "reason": "..."}},
    ...
  ]
}}

Focus on ensuring diverse themes - boost underrepresented themes."""

        try:
            response = self.llm.generate_json(prompt)
            evaluations = response.get("evaluations", [])

            for eval_item in evaluations:
                idx = eval_item.get("index")
                if idx is not None and 0 <= idx < len(activated):
                    relevance = eval_item.get("relevance", 5)
                    theme_app = eval_item.get("theme_appropriate", 5)

                    # 计算调整后的 priority
                    original = activated[idx].priority
                    llm_score = (relevance + theme_app) / 2  # 0-10

                    # 降低高占比主题的 priority，提升低占比主题的 priority
                    activated[idx].priority = int(original * 0.4 + llm_score * 0.6)

            logger.info("Rebalanced %d criteria with LLM", len(evaluations))

        except Exception as e:
            logger.warning("LLM rebalancing failed: %s", e)

        return activated

    def get_themes(self, activated: list[ActivatedCriterion]) -> list[str]:
        """获取激活的主题列表"""
        themes = [c.theme for c in activated]
        # Sort by count
        theme_counts = Counter(themes)
        return sorted(theme_counts.keys(), key=lambda t: theme_counts[t], reverse=True)

    def get_criteria_for_theme(self, activated: list[ActivatedCriterion], theme: str) -> list[ActivatedCriterion]:
        """获取特定主题的标准"""
        return [c for c in activated if c.theme == theme]

    def to_criterion_list(self, activated: list[ActivatedCriterion]) -> list[Criterion]:
        """转换为 Criterion 列表（用于向后兼容）"""
        criteria: list[Criterion] = []
        for i, ac in enumerate(activated):
            criteria.append(Criterion(
                criterion_id=f"activated_{i}",
                text=ac.criterion,
                theme=ac.theme,
                kind="content" if ac.source == "mined" else "policy",
                source_ids=[ac.trigger_reason],
            ))
        return criteria