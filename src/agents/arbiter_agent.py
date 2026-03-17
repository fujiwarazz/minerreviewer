from __future__ import annotations

import logging

from agents.base import AgentConfig
from common.types import ArbiterOutput, Criterion, ThemeOutput, VenuePolicy

logger = logging.getLogger(__name__)

# ICLR Rating Scale Reference
ICLR_RATING_SCALE = """
ICLR Rating Scale (1-10):
- 10: Top 5% of accepted papers, seminal paper (Accept)
- 9: Top 15% of accepted papers, strong accept (Accept)
- 8: Top 50% of accepted papers, clear accept (Accept)
- 7: Good paper, accept (Accept)
- 6: Marginally above acceptance threshold (Accept/Borderline)
- 5: Marginally below acceptance threshold (Borderline/Reject)
- 4: Ok but not good enough - rejection (Reject)
- 3: Clear rejection (Reject)
- 2: Strong rejection (Reject)
- 1: Trivial or wrong (Reject)

Decision categories:
- Accept: Strong contribution, minor issues only
- Borderline: Reasonable contribution but significant concerns
- Reject: Major flaws, insufficient contribution, or critical issues
- Revise: Good potential but needs substantial revision
"""


class ArbiterAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def merge(self, theme_outputs: list[ThemeOutput], policy_criteria: list[Criterion], venue_policy: VenuePolicy | None) -> ArbiterOutput:
        prompt = self._prompt(theme_outputs, policy_criteria, venue_policy)
        response = self.config.llm.generate_json(prompt)
        strengths = response.get("strengths")
        weaknesses = response.get("weaknesses")
        raw_rating = response.get("raw_rating")
        acceptance = self._parse_float(response.get("acceptance_likelihood"))
        if strengths is None or weaknesses is None:
            strengths, weaknesses, raw_rating = self._fallback(theme_outputs)
        trace = {
            "criteria_used": [c.criterion_id for c in policy_criteria],
            "policy_present": venue_policy is not None,
            "rating_rationale": response.get("rating_rationale"),
        }
        return ArbiterOutput(
            strengths=strengths,
            weaknesses=weaknesses,
            raw_rating=float(raw_rating) if raw_rating is not None else 0.0,
            decision_recommendation=response.get("decision_recommendation"),
            acceptance_likelihood=acceptance,
            trace=trace,
        )

    def _prompt(self, theme_outputs: list[ThemeOutput], policy_criteria: list[Criterion], venue_policy: VenuePolicy | None) -> str:
        serialized = [output.model_dump() for output in theme_outputs]
        policy_data = [c.model_dump() for c in policy_criteria]
        policy_meta = venue_policy.model_dump() if venue_policy else {}

        # Count strengths and weaknesses by theme
        total_strengths = sum(len(o.strengths) for o in theme_outputs)
        total_weaknesses = sum(len(o.weaknesses) for o in theme_outputs)
        severity_summary = {}
        for output in theme_outputs:
            for tag in output.severity_tags:
                severity_summary[tag] = severity_summary.get(tag, 0) + 1

        return "\n".join([
            "You are an experienced ICLR Area Chair synthesizing multiple theme reviews into a final decision.",
            "",
            "## Rating Scale Reference",
            ICLR_RATING_SCALE,
            "",
            "## Theme Review Summaries",
            f"Total themes reviewed: {len(theme_outputs)}",
            f"Total strengths identified: {total_strengths}",
            f"Total weaknesses identified: {total_weaknesses}",
            f"Severity breakdown: {severity_summary or 'None specified'}",
            "",
            "## Detailed Theme Outputs",
            *[f"### Theme: {o.theme}\nStrengths: {o.strengths}\nWeaknesses: {o.weaknesses}" for o in theme_outputs],
            "",
            "## Policy Criteria (Review Standards)",
            *[f"- {c['text']}" for c in policy_data],
            "",
            "## Venue Policy Context",
            f"{policy_meta}",
            "",
            "## Your Task",
            "Synthesize the above into a final ICLR review. You must:",
            "1. Consolidate strengths (deduplicate, prioritize by impact)",
            "2. Consolidate weaknesses (prioritize by severity and novelty concerns)",
            "3. Assign a rating (1-10) with explicit reasoning",
            "4. Recommend a decision (Accept/Reject/Borderline/Revise)",
            "",
            "## Rating Decision Logic",
            "- If multiple critical weaknesses (severity='critical'): rating <= 4, Reject",
            "- If many strengths and only minor weaknesses: rating >= 7, Accept",
            "- If balanced strengths/weaknesses with no critical issues: rating 5-6, Borderline",
            "- If good paper but needs substantial revision: rating 5-6, Revise",
            "",
            "## Output Format",
            "Return JSON with:",
            "- strengths: list of 3-6 consolidated strength statements",
            "- weaknesses: list of 3-6 consolidated weakness statements",
            "- raw_rating: float 1.0-10.0",
            "- decision_recommendation: one of 'Accept', 'Reject', 'Borderline', 'Revise'",
            "- rating_rationale: brief explanation of why this rating was chosen",
            "- acceptance_likelihood: float 0.0-1.0 (probability of acceptance given this review)",
        ])

    @staticmethod
    def _fallback(theme_outputs: list[ThemeOutput]) -> tuple[list[str], list[str], float]:
        strengths = [item for output in theme_outputs for item in output.strengths]
        weaknesses = [item for output in theme_outputs for item in output.weaknesses]
        raw_rating = 5.0 if strengths else 3.0
        return strengths, weaknesses, raw_rating

    @staticmethod
    def _parse_float(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None
