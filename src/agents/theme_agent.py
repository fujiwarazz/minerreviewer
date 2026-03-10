from __future__ import annotations

import logging

from agents.base import AgentConfig
from common.types import Criterion, Paper, ThemeOutput

logger = logging.getLogger(__name__)


class ThemeAgent:
    def __init__(self, config: AgentConfig, theme: str, use_fulltext: bool = False, max_fulltext_chars: int = 12000) -> None:
        self.config = config
        self.theme = theme
        self.use_fulltext = use_fulltext
        self.max_fulltext_chars = max_fulltext_chars

    def review(self, target: Paper, criteria: list[Criterion]) -> ThemeOutput:
        if not criteria:
            return ThemeOutput(theme=self.theme, strengths=[], weaknesses=[], severity_tags=[], criteria_used=[])
        prompt = self._prompt(target, criteria)
        response = self.config.llm.generate_json(prompt)
        strengths = response.get("strengths", [])
        weaknesses = response.get("weaknesses", [])
        severity_tags = response.get("severity_tags", [])
        if not isinstance(severity_tags, list):
            severity_tags = []
        severity_tags = [self._normalize_tag(tag) for tag in severity_tags]
        notes_value = response.get("notes")
        notes = None
        if isinstance(notes_value, str):
            notes = notes_value
        elif isinstance(notes_value, list):
            notes = " ".join(str(item) for item in notes_value)
        elif notes_value is not None:
            notes = str(notes_value)
        return ThemeOutput(
            theme=self.theme,
            strengths=strengths,
            weaknesses=weaknesses,
            severity_tags=severity_tags,
            notes=notes,
            criteria_used=[c.criterion_id for c in criteria],
        )

    def _prompt(self, target: Paper, criteria: list[Criterion]) -> str:
        serialized = [c.model_dump() for c in criteria]
        fulltext = ""
        if self.use_fulltext and target.fulltext:
            trimmed = target.fulltext[: self.max_fulltext_chars]
            fulltext = f"\nFull text (truncated):\n{trimmed}"
        return "\n".join(
            [
                f"You are the {self.theme} review agent for ICLR-style reviews.",
                f"Target paper: {target.title}\n{target.abstract}{fulltext}",
                f"Criteria: {serialized}",
                "Return JSON with strengths, weaknesses, severity_tags (aligned with weaknesses), notes.",
            ]
        )

    @staticmethod
    def _normalize_tag(tag: object) -> str:
        if isinstance(tag, str):
            return tag
        if isinstance(tag, dict):
            value = tag.get("tag") or tag.get("severity") or tag.get("value")
            if value:
                return str(value)
        if tag is None:
            return ""
        return str(tag)
