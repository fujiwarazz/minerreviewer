"""DecisionVerifier: 决策验证"""
from __future__ import annotations

import logging

from clients.llm_client import LLMClient
from common.types import ArbiterOutput, DecisionVerificationReport, Paper, RetrievalBundle

logger = logging.getLogger(__name__)


class DecisionVerifier:
    """决策验证器

    检查：
    - score-text alignment: 评分与文本描述是否一致
    - evidence support: 证据是否支持结论
    - venue alignment: 是否符合 venue 的审稿标准
    - critique specificity: 批评是否具体
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def verify(
        self,
        arbiter_output: ArbiterOutput,
        paper: Paper,
        bundle: RetrievalBundle,
    ) -> DecisionVerificationReport:
        """
        验证决策

        Args:
            arbiter_output: Arbiter 的输出
            paper: 目标论文
            bundle: 检索结果

        Returns:
            DecisionVerificationReport
        """
        # Check score-text alignment
        score_text_alignment = self._check_score_text_alignment(arbiter_output)

        # Check evidence support
        evidence_support = self._check_evidence_support(arbiter_output)

        # Check venue alignment
        venue_alignment = self._check_venue_alignment(arbiter_output, bundle)

        # Check critique specificity
        critique_warnings = self._check_critique_specificity(arbiter_output)

        # Aggregate warnings
        warnings: list[str] = []
        if score_text_alignment == "misaligned":
            warnings.append("Score and text description are misaligned")
        if evidence_support == "weak":
            warnings.append("Weak evidence support for the conclusion")
        if venue_alignment == "low":
            warnings.append("Decision may not align with venue standards")
        warnings.extend(critique_warnings)

        # Determine overall pass/fail
        passed = (
            score_text_alignment != "misaligned" and
            evidence_support != "weak" and
            len(critique_warnings) == 0
        )

        requires_revision = (
            score_text_alignment == "misaligned" or
            evidence_support == "weak" or
            venue_alignment == "low"
        )

        return DecisionVerificationReport(
            passed=passed,
            score_text_alignment=score_text_alignment,
            evidence_support_level=evidence_support,
            venue_alignment_level=venue_alignment,
            warnings=warnings,
            requires_revision=requires_revision,
        )

    def _check_score_text_alignment(self, arbiter_output: ArbiterOutput) -> str:
        """检查评分与文本是否一致"""
        rating = arbiter_output.raw_rating
        strengths = len(arbiter_output.strengths)
        weaknesses = len(arbiter_output.weaknesses)
        decision = arbiter_output.decision_recommendation or ""

        # Simple heuristic-based check
        # Lower rating should have more weaknesses
        # Higher rating should have more strengths
        if rating >= 7.0:
            # Should be accept with strong strengths
            if strengths >= 3 and weaknesses <= 2:
                if "accept" in decision.lower():
                    return "aligned"
            return "unclear"
        elif rating >= 5.0:
            # Borderline territory
            return "unclear"
        else:
            # Should be reject with significant weaknesses
            if weaknesses >= 2:
                if "reject" in decision.lower():
                    return "aligned"
                elif "borderline" in decision.lower():
                    return "unclear"
            return "misaligned"

    def _check_evidence_support(self, arbiter_output: ArbiterOutput) -> str:
        """检查证据是否支持结论"""
        strengths = arbiter_output.strengths
        weaknesses = arbiter_output.weaknesses
        decision = arbiter_output.decision_recommendation or ""

        # Check if strengths/weaknesses contain specific evidence
        def has_evidence(items: list[str]) -> int:
            count = 0
            evidence_keywords = ["experiment", "result", "data", "table", "figure", "baseline", "comparison", "analysis", "proof"]
            for item in items:
                if any(kw in item.lower() for kw in evidence_keywords):
                    count += 1
            return count

        strength_evidence = has_evidence(strengths)
        weakness_evidence = has_evidence(weaknesses)
        total_evidence = strength_evidence + weakness_evidence

        if total_evidence >= 3:
            return "strong"
        elif total_evidence >= 1:
            return "moderate"
        else:
            return "weak"

    def _check_venue_alignment(self, arbiter_output: ArbiterOutput, bundle: RetrievalBundle) -> str:
        """检查是否符合 venue 审稿标准"""
        # Check against policy cards
        policy_cards = bundle.policy_cards
        if not policy_cards:
            return "unknown"

        # Check if decision aligns with policy patterns
        decision = arbiter_output.decision_recommendation or ""
        rating = arbiter_output.raw_rating

        # Simple heuristic: check if similar cases support this decision
        similar_cases = bundle.similar_paper_cases
        if not similar_cases:
            return "unknown"

        # Count similar decisions (case-insensitive)
        accept_count = sum(1 for c in similar_cases if c.decision and "accept" in c.decision.lower())
        reject_count = sum(1 for c in similar_cases if c.decision and "reject" in c.decision.lower())

        if "accept" in decision.lower() and accept_count > reject_count:
            return "high"
        elif "reject" in decision.lower() and reject_count > accept_count:
            return "high"
        elif "borderline" in decision.lower():
            return "medium"
        else:
            return "low"

    def _check_critique_specificity(self, arbiter_output: ArbiterOutput) -> list[str]:
        """检查批评是否具体"""
        warnings: list[str] = []
        weaknesses = arbiter_output.weaknesses

        # Check for vague criticisms
        vague_patterns = [
            ("unclear", "Unclear criticism lacks specificity"),
            ("not clear", "Vague criticism should be more specific"),
            ("problematic", "Problematic issue needs more detail"),
            ("issue", "Issue should be elaborated"),
            ("concern", "Concern should be more specific"),
        ]

        for weakness in weaknesses:
            weakness_lower = weakness.lower()
            for pattern, message in vague_patterns:
                if pattern in weakness_lower and len(weakness) < 50:
                    # Short and vague criticism
                    warnings.append(message)
                    break

        return warnings

    def verify_with_llm(
        self,
        arbiter_output: ArbiterOutput,
        paper: Paper,
    ) -> DecisionVerificationReport:
        """使用 LLM 进行更详细的验证"""
        prompt = self._build_verification_prompt(arbiter_output, paper)
        try:
            response = self.llm.generate_json(prompt)
            return DecisionVerificationReport(
                passed=response.get("passed", True),
                score_text_alignment=response.get("score_text_alignment", "unclear"),
                evidence_support_level=response.get("evidence_support_level", "moderate"),
                venue_alignment_level=response.get("venue_alignment_level", "unknown"),
                warnings=response.get("warnings", []),
                requires_revision=response.get("requires_revision", False),
            )
        except Exception as e:
            logger.warning("LLM verification failed: %s", e)
            return self.verify(arbiter_output, paper, RetrievalBundle(target_paper=paper))

    def _build_verification_prompt(self, arbiter_output: ArbiterOutput, paper: Paper) -> str:
        return "\n".join([
            "Verify the following review decision for consistency and quality.",
            "",
            f"Paper: {paper.title}",
            f"Abstract: {paper.abstract[:500]}...",
            "",
            f"Generated Rating: {arbiter_output.raw_rating}",
            f"Decision: {arbiter_output.decision_recommendation}",
            "",
            f"Strengths: {arbiter_output.strengths}",
            f"Weaknesses: {arbiter_output.weaknesses}",
            "",
            "Check the following and return JSON:",
            "- passed (bool): Does the review pass basic quality checks?",
            "- score_text_alignment (aligned/misaligned/unclear): Does rating match strengths/weaknesses?",
            "- evidence_support_level (strong/moderate/weak): Are claims supported by evidence?",
            "- venue_alignment_level (high/medium/low/unknown): Does it match venue standards?",
            "- warnings (list[str]): Any issues found",
            "- requires_revision (bool): Should the review be revised?",
        ])