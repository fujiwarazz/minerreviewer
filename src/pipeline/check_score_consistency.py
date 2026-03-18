"""ScoreConsistencyChecker: 评分一致性检查（只警告，不改分）"""
from __future__ import annotations

import logging
import statistics

from common.types import ArbiterOutput, PaperCase, RetrievalBundle, Review, ScoreConsistencyReport

logger = logging.getLogger(__name__)


class ScoreConsistencyChecker:
    """评分一致性检查器

    优先使用 similar_paper_cases 作为一致性参考来源，
    只提供警告和 justification_needed 标志，不修改 raw_rating 和 decision_recommendation
    """

    def __init__(
        self,
        rating_tolerance: float = 1.5,
        deviation_threshold: float = 2.0,
        min_samples: int = 3,
        prefer_cases: bool = True,  # 优先使用 cases
    ) -> None:
        self.rating_tolerance = rating_tolerance
        self.deviation_threshold = deviation_threshold
        self.min_samples = min_samples
        self.prefer_cases = prefer_cases

    def check(
        self,
        arbiter_output: ArbiterOutput,
        bundle: RetrievalBundle,
    ) -> ScoreConsistencyReport:
        """
        检查评分一致性

        知识来源优先级：
        1. similar_paper_cases (case-level stats)
        2. related_reviews (fallback, legacy)

        Args:
            arbiter_output: Arbiter 的输出
            bundle: 检索结果（包含 similar_paper_cases 和 related_reviews）

        Returns:
            ScoreConsistencyReport
        """
        # 优先使用 similar_paper_cases
        if self.prefer_cases and bundle.similar_paper_cases:
            return self._check_with_cases(arbiter_output, bundle)

        # Fallback to related_reviews
        return self._check_with_reviews(arbiter_output, bundle)

    def _check_with_cases(
        self,
        arbiter_output: ArbiterOutput,
        bundle: RetrievalBundle,
    ) -> ScoreConsistencyReport:
        """使用 similar_paper_cases 进行一致性检查"""
        cases = bundle.similar_paper_cases
        rated_cases = [c for c in cases if c.rating is not None]

        if len(rated_cases) < self.min_samples:
            # Fallback to reviews if not enough cases
            logger.info(
                "Not enough rated cases (%d), falling back to reviews",
                len(rated_cases)
            )
            return self._check_with_reviews(arbiter_output, bundle)

        # Calculate statistics from cases
        ratings = [c.rating for c in rated_cases if c.rating is not None]
        mean_rating = statistics.mean(ratings)
        median_rating = statistics.median(ratings)
        rating_deviation = statistics.stdev(ratings) if len(ratings) > 1 else 0.0

        # Decision distribution from cases
        decision_distribution: dict[str, int] = {}
        for case in rated_cases:
            if case.decision:
                decision_key = self._normalize_decision(case.decision)
                decision_distribution[decision_key] = decision_distribution.get(decision_key, 0) + 1

        # Check consistency
        raw_rating = arbiter_output.raw_rating
        deviation_from_mean = abs(raw_rating - mean_rating)

        consistency_level = "high"
        warning = None
        justification_needed = False

        if deviation_from_mean > self.deviation_threshold:
            consistency_level = "low"
            warning = (
                f"Rating {raw_rating:.1f} deviates {deviation_from_mean:.1f} from "
                f"similar cases mean {mean_rating:.1f} (based on {len(rated_cases)} cases)"
            )
            justification_needed = True
        elif deviation_from_mean > self.rating_tolerance:
            consistency_level = "medium"
            warning = (
                f"Rating {raw_rating:.1f} somewhat deviates from similar cases "
                f"(mean={mean_rating:.1f}, n={len(rated_cases)})"
            )

        # Check decision alignment with cases
        if decision_distribution:
            majority_decision = max(decision_distribution.items(), key=lambda x: x[1])[0]
            arbiter_decision = arbiter_output.decision_recommendation or ""
            arbiter_decision_normalized = self._normalize_decision(arbiter_decision)

            if majority_decision != arbiter_decision_normalized:
                if warning:
                    warning += f". Decision '{arbiter_decision}' differs from case majority '{majority_decision}'"
                else:
                    warning = (
                        f"Decision '{arbiter_decision}' differs from case majority "
                        f"'{majority_decision}' (based on {len(rated_cases)} similar cases)"
                    )
                justification_needed = True

        return ScoreConsistencyReport(
            similar_review_count=len(rated_cases),
            mean_rating=mean_rating,
            median_rating=median_rating,
            rating_deviation=rating_deviation,
            decision_distribution=decision_distribution,
            consistency_level=consistency_level,
            warning=warning,
            justification_needed=justification_needed,
        )

    def _check_with_reviews(
        self,
        arbiter_output: ArbiterOutput,
        bundle: RetrievalBundle,
    ) -> ScoreConsistencyReport:
        """使用 related_reviews 进行一致性检查 (fallback)"""
        # Collect similar reviews with ratings
        similar_reviews = bundle.related_reviews
        rated_reviews = [r for r in similar_reviews if r.rating is not None]

        if len(rated_reviews) < self.min_samples:
            return ScoreConsistencyReport(
                similar_review_count=len(rated_reviews),
                consistency_level="unknown",
                warning=f"Insufficient similar reviews for consistency check (need {self.min_samples}, got {len(rated_reviews)})",
                justification_needed=False,
            )

        # Calculate statistics
        ratings = [r.rating for r in rated_reviews if r.rating is not None]
        mean_rating = statistics.mean(ratings)
        median_rating = statistics.median(ratings)
        rating_deviation = statistics.stdev(ratings) if len(ratings) > 1 else 0.0

        # Decision distribution
        decision_distribution: dict[str, int] = {}
        for r in rated_reviews:
            if r.decision:
                decision_key = self._normalize_decision(r.decision)
                decision_distribution[decision_key] = decision_distribution.get(decision_key, 0) + 1

        # Check consistency
        raw_rating = arbiter_output.raw_rating
        deviation_from_mean = abs(raw_rating - mean_rating)

        consistency_level = "high"
        warning = None
        justification_needed = False

        if deviation_from_mean > self.deviation_threshold:
            consistency_level = "low"
            warning = f"Rating {raw_rating:.1f} deviates {deviation_from_mean:.1f} from mean {mean_rating:.1f}"
            justification_needed = True
        elif deviation_from_mean > self.rating_tolerance:
            consistency_level = "medium"
            warning = f"Rating {raw_rating:.1f} somewhat deviates from similar reviews (mean={mean_rating:.1f})"

        # Check decision alignment
        if decision_distribution:
            majority_decision = max(decision_distribution.items(), key=lambda x: x[1])[0]
            arbiter_decision = arbiter_output.decision_recommendation or ""
            arbiter_decision_normalized = self._normalize_decision(arbiter_decision)

            if majority_decision != arbiter_decision_normalized:
                if warning:
                    warning += f". Decision '{arbiter_decision}' differs from majority '{majority_decision}'"
                else:
                    warning = f"Decision '{arbiter_decision}' differs from majority '{majority_decision}' among similar reviews"
                justification_needed = True

        return ScoreConsistencyReport(
            similar_review_count=len(rated_reviews),
            mean_rating=mean_rating,
            median_rating=median_rating,
            rating_deviation=rating_deviation,
            decision_distribution=decision_distribution,
            consistency_level=consistency_level,
            warning=warning,
            justification_needed=justification_needed,
        )

    def _normalize_decision(self, decision: str) -> str:
        """标准化决策字符串"""
        decision_lower = decision.lower()
        if "accept" in decision_lower:
            return "accept"
        elif "reject" in decision_lower:
            return "reject"
        elif "borderline" in decision_lower:
            return "borderline"
        elif "revise" in decision_lower:
            return "revise"
        return "unknown"