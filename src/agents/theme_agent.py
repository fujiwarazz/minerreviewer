from __future__ import annotations

import logging

from agents.base import AgentConfig
from common.types import Criterion, Paper, ThemeOutput

logger = logging.getLogger(__name__)

# Few-shot examples for review writing
REVIEW_EXAMPLES = """
## Example Good Strength:
"The paper proposes a novel attention mechanism that reduces computational complexity from O(n²) to O(n log n) while maintaining comparable accuracy on language modeling tasks. The theoretical analysis in Section 3 provides clear justification for the approximation bounds."

## Example Good Weakness:
"The empirical evaluation is limited to only two datasets (WikiText-2 and PTB). The paper does not evaluate on larger-scale benchmarks like LAMBADA or Wikitext-103, which makes it difficult to assess scalability. Additionally, no comparison is provided against recent efficient attention variants (e.g., Linformer, Performer)."

## Example Bad Strength (too vague):
"The paper is well written and the method is interesting."

## Example Bad Weakness (no evidence):
"The experiments are not convincing."
"""


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
            fulltext = f"\n## Paper Full Text (truncated):\n{trimmed}"

        criteria_list = "\n".join([
            f"{i+1}. [{c.theme}] {c.text}"
            for i, c in enumerate(criteria)
        ])

        return "\n".join([
            f"You are an expert reviewer focusing on the '{self.theme}' aspect of an ICLR paper submission.",
            "",
            "## Your Role",
            f"Evaluate the paper specifically on criteria related to '{self.theme}'.",
            "Provide specific, evidence-based comments that would help the authors improve their paper.",
            "",
            "## Review Quality Standards",
            "- Every claim must cite specific evidence (section numbers, equations, experiments, tables)",
            "- Weaknesses should suggest concrete improvements when possible",
            "- Avoid vague statements; be specific and actionable",
            "- Balance depth and breadth; cover all criteria but with substantive analysis",
            "",
            REVIEW_EXAMPLES,
            "",
            "## Target Paper",
            f"Title: {target.title}",
            f"Abstract: {target.abstract}",
            fulltext,
            "",
            "## Criteria to Evaluate",
            criteria_list,
            "",
            "## Output Requirements",
            "Return JSON with:",
            "- strengths: list of 2-4 specific strengths with evidence citations",
            "- weaknesses: list of 2-4 specific weaknesses with evidence citations",
            "- severity_tags: list of severity levels corresponding to each weakness",
            "  - 'critical': fundamental flaw that warrants rejection",
            "  - 'major': significant issue that should be addressed",
            "  - 'minor': small issue that could be easily fixed",
            "- notes: optional additional comments or suggestions",
            "",
            "Remember: Quality > Quantity. Each point should be substantive and well-evidenced.",
        ])

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
