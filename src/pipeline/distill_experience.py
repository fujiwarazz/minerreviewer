"""ExperienceDistiller: 从审稿轨迹中提取经验"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from clients.llm_client import LLMClient
from common.types import ArbiterOutput, ExperienceCard, Paper, PaperCase, PaperSignature, Review, RetrievalBundle

logger = logging.getLogger(__name__)


class ExperienceDistiller:
    """从审稿轨迹中提取经验"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def distill(
        self,
        arbiter_output: ArbiterOutput,
        paper: Paper,
        signature: PaperSignature | None,
        bundle: RetrievalBundle,
        target_year: int | None = None,
    ) -> DistillationResult:
        """
        从审稿轨迹中提取经验

        Args:
            arbiter_output: Arbiter 的输出
            paper: 目标论文
            signature: 论文结构化特征
            bundle: 检索结果
            target_year: 目标年份

        Returns:
            DistillationResult 包含提取的各类经验
        """
        result = DistillationResult()

        # 1. Extract paper case
        paper_case = self._extract_paper_case(arbiter_output, paper, signature)
        result.paper_case = paper_case

        # 2. Extract policy updates from successful criteria
        policy_updates = self._extract_policy_updates(arbiter_output, bundle)
        result.policy_updates = policy_updates

        # 3. Extract critique cases from weaknesses
        critique_cases = self._extract_critique_cases(arbiter_output, paper)
        result.critique_cases = critique_cases

        # 4. Extract failure patterns if applicable
        if self._is_failure_case(arbiter_output):
            failure_cards = self._extract_failure_patterns(arbiter_output, paper)
            result.failure_cards = failure_cards

        return result

    def _extract_paper_case(
        self,
        arbiter_output: ArbiterOutput,
        paper: Paper,
        signature: PaperSignature | None,
    ) -> PaperCase:
        """提取论文案例"""
        return PaperCase(
            case_id=str(uuid.uuid4()),
            paper_id=paper.paper_id,
            venue_id=paper.venue_id,
            year=paper.year,
            title=paper.title,
            abstract=paper.abstract,
            paper_signature=signature,
            top_strengths=arbiter_output.strengths[:5],
            top_weaknesses=arbiter_output.weaknesses[:5],
            decisive_issues=self._extract_decisive_issues(arbiter_output),
            decision=arbiter_output.decision_recommendation,
            rating=arbiter_output.raw_rating,
            source_review_ids=[],
            transferable_criteria=self._extract_transferable_criteria(arbiter_output),
            failure_patterns=[],
        )

    def _extract_decisive_issues(self, arbiter_output: ArbiterOutput) -> list[str]:
        """提取决定性问题"""
        issues: list[str] = []
        for weakness in arbiter_output.weaknesses:
            # Look for critical keywords
            if any(kw in weakness.lower() for kw in ["critical", "major", "fundamental", "severe"]):
                issues.append(weakness)
        return issues[:3]

    def _extract_transferable_criteria(self, arbiter_output: ArbiterOutput) -> list[str]:
        """提取可迁移的审稿标准"""
        criteria: list[str] = []
        for weakness in arbiter_output.weaknesses:
            if len(weakness) > 20 and any(kw in weakness.lower() for kw in ["should", "could", "would", "needs", "lacks"]):
                criteria.append(weakness[:200])
        return list(set(criteria))[:5]

    def _extract_policy_updates(
        self,
        arbiter_output: ArbiterOutput,
        bundle: RetrievalBundle,
    ) -> list[ExperienceCard]:
        """提取策略更新"""
        cards: list[ExperienceCard] = []

        # Extract from strengths for high-quality patterns
        for strength in arbiter_output.strengths:
            if len(strength) > 30:
                card = ExperienceCard(
                    card_id=str(uuid.uuid4()),
                    kind="policy",
                    scope="venue",
                    venue_id=bundle.target_paper.venue_id,
                    theme="quality",
                    content=strength,
                    utility=arbiter_output.raw_rating / 10.0,
                    confidence=0.5,
                    source_ids=[],
                )
                cards.append(card)

        return cards[:3]

    def _extract_critique_cases(
        self,
        arbiter_output: ArbiterOutput,
        paper: Paper,
    ) -> list[ExperienceCard]:
        """提取批评案例"""
        cards: list[ExperienceCard] = []

        for weakness in arbiter_output.weaknesses:
            if len(weakness) > 30:
                # Infer theme from weakness content
                theme = self._infer_theme(weakness)
                card = ExperienceCard(
                    card_id=str(uuid.uuid4()),
                    kind="critique",
                    scope="venue",
                    venue_id=paper.venue_id,
                    theme=theme,
                    content=weakness,
                    trigger=[],
                    utility=0.5,
                    confidence=0.5,
                    source_ids=[],
                )
                cards.append(card)

        return cards[:3]

    def _is_failure_case(self, arbiter_output: ArbiterOutput) -> bool:
        """判断是否为失败案例"""
        decision = arbiter_output.decision_recommendation or ""
        return "reject" in decision.lower() or arbiter_output.raw_rating < 5.0

    def _extract_failure_patterns(
        self,
        arbiter_output: ArbiterOutput,
        paper: Paper,
    ) -> list[ExperienceCard]:
        """提取失败模式"""
        cards: list[ExperienceCard] = []

        for weakness in arbiter_output.weaknesses:
            if any(kw in weakness.lower() for kw in ["missing", "lacks", "insufficient", "incomplete", "unclear"]):
                theme = self._infer_theme(weakness)
                card = ExperienceCard(
                    card_id=str(uuid.uuid4()),
                    kind="failure",
                    scope="venue",
                    venue_id=paper.venue_id,
                    theme=theme,
                    content=weakness,
                    trigger=[],
                    utility=0.3,  # Low utility for failure patterns
                    confidence=0.5,
                    source_ids=[],
                )
                cards.append(card)

        return cards[:3]

    def _infer_theme(self, text: str) -> str:
        """推断主题"""
        text_lower = text.lower()
        theme_keywords = {
            "novelty": ["novel", "new", "original", "contribution"],
            "quality": ["quality", "correctness", "accuracy", "performance"],
            "clarity": ["clarity", "writing", "presentation", "explain"],
            "significance": ["significant", "impact", "important"],
            "reproducibility": ["reproducib", "code", "implementation", "detail"],
            "soundness": ["sound", "theory", "proof", "mathematical"],
        }
        for theme, keywords in theme_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return theme
        return "quality"


class DistillationResult:
    """蒸馏结果"""

    def __init__(self) -> None:
        self.paper_case: PaperCase | None = None
        self.policy_updates: list[ExperienceCard] = []
        self.critique_cases: list[ExperienceCard] = []
        self.failure_cards: list[ExperienceCard] = []

    def has_updates(self) -> bool:
        return bool(
            self.paper_case or
            self.policy_updates or
            self.critique_cases or
            self.failure_cards
        )

    def all_cards(self) -> list[ExperienceCard]:
        """获取所有卡片"""
        cards: list[ExperienceCard] = []
        cards.extend(self.policy_updates)
        cards.extend(self.critique_cases)
        cards.extend(self.failure_cards)
        return cards