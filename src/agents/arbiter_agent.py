from __future__ import annotations

import logging

from agents.base import AgentConfig
from common.types import ArbiterOutput, Criterion, ThemeOutput, VenuePolicy

logger = logging.getLogger(__name__)


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
        return "\n".join(
            [
                "You are the meta-reviewer consolidating ICLR-style reviews.",
                f"Theme outputs: {serialized}",
                f"Policy criteria: {policy_data}",
                f"Venue policy: {policy_meta}",
                "Return JSON with strengths, weaknesses, decision_recommendation.",
            ]
        )

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
