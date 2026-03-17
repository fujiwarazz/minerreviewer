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
        max_criteria: int = 15,
        priority_boost_memory: float = 1.5,
    ) -> list[ActivatedCriterion]:
        """
        规划审稿标准，返回激活的标准列表

        Args:
            signature: 论文结构化特征
            bundle: 多通道检索结果
            mined_criteria: 从历史数据挖掘的标准
            max_criteria: 最大标准数量
            priority_boost_memory: 记忆来源的优先级加成

        Returns:
            激活的标准列表，按优先级排序
        """
        activated: list[ActivatedCriterion] = []

        # 1. Add criteria from policy cards (memory)
        for card in bundle.policy_cards:
            priority = int(card.utility * 10 * priority_boost_memory)
            activated.append(ActivatedCriterion(
                theme=card.theme,
                criterion=card.content,
                source="memory",
                priority=priority,
                trigger_reason=f"Retrieved from memory with utility={card.utility:.2f}",
                required_evidence=card.trigger,
                owner_agent=f"theme_{card.theme}",
            ))
            # Update use count
            card.use_count += 1

        # 2. Add criteria from similar paper cases
        for case in bundle.similar_paper_cases:
            for criterion_text in case.transferable_criteria:
                # Determine theme based on content
                theme = self._infer_theme(criterion_text)
                priority = 5  # Base priority for case-based criteria
                if case.decision == "reject":
                    priority += 2  # Boost for rejected papers
                activated.append(ActivatedCriterion(
                    theme=theme,
                    criterion=criterion_text,
                    source="memory",
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
                source="memory",
                priority=8,  # High priority for failure patterns
                trigger_reason=f"Failure pattern: {card.trigger}",
                required_evidence=card.trigger,
                owner_agent=f"theme_{theme}",
            ))

        # 4. Add mined criteria
        for criterion in mined_criteria:
            priority = 3  # Base priority for mined criteria
            activated.append(ActivatedCriterion(
                theme=criterion.theme,
                criterion=criterion.text,
                source="mined",
                priority=priority,
                trigger_reason=f"Mined from {criterion.kind} criteria",
                owner_agent=f"theme_{criterion.theme}",
            ))

        # 5. Deduplicate and merge similar criteria
        activated = self._deduplicate(activated)

        # 6. Sort by priority and limit
        activated.sort(key=lambda x: x.priority, reverse=True)
        activated = activated[:max_criteria]

        logger.info("Planned %d activated criteria", len(activated))
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