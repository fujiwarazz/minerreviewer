from __future__ import annotations

from agents.arbiter_agent import ArbiterAgent
from common.types import ArbiterOutput, Criterion, PaperCase, ThemeOutput, VenuePolicy


class Aggregator:
    def __init__(self, arbiter: ArbiterAgent) -> None:
        self.arbiter = arbiter

    def aggregate(
        self,
        theme_outputs: list[ThemeOutput],
        policy_criteria: list[Criterion],
        venue_policy: VenuePolicy | None,
        similar_cases: list[PaperCase] | None = None,
    ) -> ArbiterOutput:
        return self.arbiter.merge(theme_outputs, policy_criteria, venue_policy, similar_cases)
