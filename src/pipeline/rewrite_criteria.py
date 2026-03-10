from __future__ import annotations

import logging

from clients.llm_client import LLMClient
from common.types import Criterion, Paper

logger = logging.getLogger(__name__)


class CriteriaRewriter:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def rewrite(self, target: Paper, criteria: list[Criterion]) -> list[Criterion]:
        if not criteria:
            return []
        prompt = self._prompt(target, criteria)
        response = self.llm.generate_json(prompt)
        rewritten = response.get("criteria", [])
        result: list[Criterion] = []
        for original, item in zip(criteria, rewritten):
            # Handle case where item might be a string or dict
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text", original.text)
            else:
                text = original.text
            result.append(original.model_copy(update={"text": text}))
        return result

    @staticmethod
    def _prompt(target: Paper, criteria: list[Criterion]) -> str:
        serialized = [c.model_dump() for c in criteria]
        return "\n".join(
            [
                "Rewrite the criteria to be paper-specific and evidence-binding.",
                f"Target paper: {target.title}\n{target.abstract}",
                f"Criteria: {serialized}",
                "Return JSON with key 'criteria': list of {text} aligned to input order.",
            ]
        )
