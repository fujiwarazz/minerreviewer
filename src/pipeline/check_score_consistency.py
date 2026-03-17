"""ScoreConsistencyChecker: 评分一致性检查（只警告，不改分）"""
from __future__ import annotations

import logging
import statistics

from common.types import ArbiterOutput, RetrievalBundle, Review, ScoreConsistencyReport

logger = logging.getLogger(__name__)


class ScoreConsistencyChecker:
    """评分一致性检查器

    只提供警告和 justification_needed 标志，不修改 raw_rating 和 decision_recommendation
    """

    def __init__(
        self,
        rating_tolerance: float = 1.5,
        deviation_threshold: float = 2.0,
        min_samples: int = 3,
    ) -> None:
        self.rating_tolerance = rating_tolerance
        self.deviation_threshold = deviation_threshold
        self.min_samples = min_samples

    def check(
        self,
        arbiter_output: ArbiterOutput,
        bundle: RetrievalBundle,
    ) -> ScoreConsistencyReport:
        """
        检查评分一致性

        Args:
            arbiter_output: Arbiter 的输出
            bundle: 检索结果（包含 similar reviews）

        Returns:
            ScoreConsistencyReport
        """
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