from __future__ import annotations

import logging
import statistics

from agents.base import AgentConfig
from common.types import ArbiterOutput, Criterion, PaperCase, ThemeOutput, VenuePolicy

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

    def merge(
        self,
        theme_outputs: list[ThemeOutput],
        policy_criteria: list[Criterion],
        venue_policy: VenuePolicy | None,
        similar_cases: list[PaperCase] | None = None,
    ) -> ArbiterOutput:
        prompt = self._prompt(theme_outputs, policy_criteria, venue_policy, similar_cases)
        response = self.config.llm.generate_json(prompt)
        strengths = response.get("strengths")
        weaknesses = response.get("weaknesses")
        raw_rating = response.get("raw_rating")
        acceptance = self._parse_float(response.get("acceptance_likelihood"))
        if strengths is None or weaknesses is None:
            strengths, weaknesses, raw_rating = self._fallback(theme_outputs, similar_cases)
        trace = {
            "criteria_used": [c.criterion_id for c in policy_criteria],
            "policy_present": venue_policy is not None,
            "similar_cases_used": len(similar_cases) if similar_cases else 0,
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

    def _prompt(
        self,
        theme_outputs: list[ThemeOutput],
        policy_criteria: list[Criterion],
        venue_policy: VenuePolicy | None,
        similar_cases: list[PaperCase] | None = None,
    ) -> str:
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

        # Build similar cases reference for rating calibration
        cases_section = ""
        rating_reference = ""
        rating_anchor = ""  # 新增：评分锚点，用于强调

        if similar_cases:
            valid_cases = [c for c in similar_cases if c.rating is not None]
            if valid_cases:
                ratings = [c.rating for c in valid_cases]
                mean_rating = statistics.mean(ratings)
                median_rating = statistics.median(ratings)
                accept_count = sum(1 for c in valid_cases if c.decision and "accept" in c.decision.lower())
                reject_count = sum(1 for c in valid_cases if c.decision and "reject" in c.decision.lower())

                # 计算评分范围
                min_rating = min(ratings)
                max_rating = max(ratings)

                cases_section = "\n".join([
                    "",
                    "## 🔔 SIMILAR PAPER CASES (LEARN FROM THESE)",
                    f"Found {len(valid_cases)} similar papers from historical reviews.",
                    "",
                    "Study these cases to understand the venue's review patterns:",
                    "",
                ])

                # Add each case with full details for learning
                for i, c in enumerate(valid_cases[:5]):
                    case_detail = f"""### Case {i+1}: {c.title[:50]}...
- **Rating**: {c.rating}
- **Decision**: {c.decision or 'N/A'}
- **Key Strengths**: {c.top_strengths[:2] if c.top_strengths else 'N/A'}
- **Key Weaknesses**: {c.top_weaknesses[:2] if c.top_weaknesses else 'N/A'}
- **Decisive Issues**: {c.decisive_issues[:2] if c.decisive_issues else 'N/A'}
"""
                    cases_section += case_detail

                cases_section += f"""
**Rating Distribution:** Min={min_rating:.1f}, Mean={mean_rating:.1f}, Median={median_rating:.1f}, Max={max_rating:.1f}
**Decision Distribution:** Accept={accept_count}, Reject={reject_count}

⚠️ **Learn from these cases:**
- What patterns led to Accept vs Reject?
- What issues were considered decisive?
- Apply similar reasoning to the target paper.
"""

                # 不预设阈值，让 Arbiter 自己从案例中学习
                rating_reference = f"""
═══════════════════════════════════════════════════════════════
📊 RATING ANALYSIS FROM SIMILAR CASES

**Statistics from {len(valid_cases)} similar papers:**
- Mean Rating: {mean_rating:.1f}
- Median Rating: {median_rating:.1f}
- Range: {min_rating:.1f} - {max_rating:.1f}
- Decision Distribution: {accept_count} Accept, {reject_count} Reject

**Your Task - Learn from these cases:**
1. Analyze the rating patterns above - what rating thresholds separate Accept from Reject?
2. Compare each case's strengths/weaknesses to its rating
3. Identify what quality factors led to higher/lower ratings
4. Apply the same reasoning to the target paper

**Important:**
- You should derive your own rating thresholds from the similar cases
- Don't use preset thresholds - learn from the data
- Explain your reasoning by referencing specific cases
═══════════════════════════════════════════════════════════════
"""
                rating_anchor = f"[ANCHOR: {mean_rating:.1f}]"

        return "\n".join([
            "You are an experienced ICLR Area Chair synthesizing multiple theme reviews into a final decision.",
            "",
            cases_section,
            rating_reference,
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
            "Synthesize the above into a final review. You must:",
            "1. Consolidate strengths (deduplicate, prioritize by impact)",
            "2. Consolidate weaknesses (prioritize by severity and novelty concerns)",
            "3. Assign a rating (1-10) based on patterns learned from similar cases",
            "4. Recommend a decision consistent with similar case patterns",
            "",
            "## Decision Process",
            "1. Review the similar cases above - what led to Accept vs Reject?",
            "2. Compare this paper's strengths/weaknesses to those cases",
            "3. Determine if this paper is stronger, weaker, or comparable",
            "4. Assign rating and decision accordingly",
            "",
            "## Output Format",
            "Return JSON with:",
            "- strengths: list of 3-6 consolidated strength statements",
            "- weaknesses: list of 3-6 consolidated weakness statements",
            "- raw_rating: float 1.0-10.0",
            "- decision_recommendation: one of 'Accept', 'Reject', 'Borderline', 'Revise'",
            "- rating_rationale: explain how similar cases influenced your decision",
            "- acceptance_likelihood: float 0.0-1.0",
        ])

    @staticmethod
    def _fallback(
        theme_outputs: list[ThemeOutput],
        similar_cases: list[PaperCase] | None = None,
    ) -> tuple[list[str], list[str], float]:
        strengths = [item for output in theme_outputs for item in output.strengths]
        weaknesses = [item for output in theme_outputs for item in output.weaknesses]

        # If we have similar cases, use their mean rating as baseline
        if similar_cases:
            valid_cases = [c for c in similar_cases if c.rating is not None]
            if valid_cases:
                import statistics
                base_rating = statistics.mean([c.rating for c in valid_cases])

                # More aggressive adjustment based on strengths/weaknesses balance
                diff = len(strengths) - len(weaknesses)
                if diff > 2:
                    # 明显更多 strengths
                    adjusted = min(10.0, base_rating + 0.5)
                elif diff < -2:
                    # 明显更多 weaknesses
                    adjusted = max(1.0, base_rating - 1.0)
                elif diff > 0:
                    adjusted = min(10.0, base_rating + 0.3)
                elif diff < 0:
                    adjusted = max(1.0, base_rating - 0.5)
                else:
                    adjusted = base_rating

                logger.info("Fallback: using similar cases mean=%.1f, adjusted=%.1f", base_rating, adjusted)
                return strengths, weaknesses, adjusted

        # No similar cases - use default logic
        if strengths and not weaknesses:
            raw_rating = 7.0
        elif weaknesses and not strengths:
            raw_rating = 3.0
        elif len(strengths) > len(weaknesses):
            raw_rating = 6.0
        elif len(weaknesses) > len(strengths):
            raw_rating = 4.0
        else:
            raw_rating = 5.0

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
