from __future__ import annotations

import logging

from agents.base import AgentConfig
from common.types import Criterion, ExperienceCard, Paper, ThemeOutput

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

    def review(self, target: Paper, criteria: list[Criterion], policy_cards: list[ExperienceCard] = None, critique_cards: list[ExperienceCard] = None) -> ThemeOutput:
        if not criteria and not policy_cards:
            return ThemeOutput(theme=self.theme, strengths=[], weaknesses=[], severity_tags=[], criteria_used=[])
        policy_cards = policy_cards or []
        critique_cards = critique_cards or []
        prompt = self._prompt(target, criteria, policy_cards, critique_cards)
        response = self.config.llm.generate_json(prompt)

        # 处理 LLM 返回 list 格式的情况
        if isinstance(response, list):
            # 尝试从 list 中提取内容
            strengths = []
            weaknesses = []
            for item in response:
                if isinstance(item, dict):
                    if "strength" in item or "point" in item:
                        strengths.append(item)
                    elif "weakness" in item or "issue" in item:
                        weaknesses.append(item)
            logger.warning(f"[{self.theme}] LLM returned list format, extracted {len(strengths)} strengths, {len(weaknesses)} weaknesses")
            return ThemeOutput(
                theme=self.theme,
                strengths=strengths,
                weaknesses=weaknesses,
                severity_tags=[],
                notes=None,
                criteria_used=[c.criterion_id for c in criteria],
            )

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

    def _prompt(self, target: Paper, criteria: list[Criterion], policy_cards: list[ExperienceCard] = None, critique_cards: list[ExperienceCard] = None) -> str:
        serialized = [c.model_dump() for c in criteria]
        fulltext = ""
        if self.use_fulltext and target.fulltext:
            trimmed = target.fulltext[: self.max_fulltext_chars]
            fulltext = f"\n## Paper Full Text (truncated):\n{trimmed}"

        criteria_list = "\n".join([
            f"{i+1}. [{c.theme}] {c.text}"
            for i, c in enumerate(criteria)
        ])

        # Add policy cards section
        policy_cards = policy_cards or []
        policy_section = ""
        if policy_cards:
            policy_items = []
            for card in policy_cards[:10]:  # 增加到 10 条
                # Handle both dict and Pydantic model
                if hasattr(card, 'content'):
                    content = card.content or ""
                else:
                    content = card.get("content", "")
                if content:
                    policy_items.append(f"- {content[:150]}")
            if policy_items:
                policy_section = "\n## Review Standards (Policy Cards)\n" + "\n".join(policy_items) + "\n"

        # Add critique cards section (criticism patterns for weaknesses)
        critique_cards = critique_cards or []
        critique_section = ""
        if critique_cards:
            critique_items = []
            for card in critique_cards[:8]:  # 最多 8 条 critique
                if hasattr(card, 'content'):
                    content = card.content or ""
                    trigger = card.trigger or []
                else:
                    content = card.get("content", "")
                    trigger = card.get("trigger", [])
                if content:
                    trigger_str = f" (trigger: {', '.join(trigger[:2])})" if trigger else ""
                    critique_items.append(f"- {content[:120]}{trigger_str}")
            if critique_items:
                critique_section = "\n## Common Criticism Patterns (Critique Cards)\nWhen identifying weaknesses, also check for these common criticism angles:\n" + "\n".join(critique_items) + "\n"

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
            policy_section,
            critique_section,
            "## Critical Evaluation Guidelines",
            "When identifying weaknesses, also consider these high-level questions:",
            "",
            "**Novelty & Originality:**",
            "- Is this merely a straightforward extension or minor modification of existing methods?",
            "- Does the paper introduce genuinely new concepts or just repackage existing ideas?",
            "- Are the contributions incremental rather than substantial advances?",
            "",
            "**Practical Applicability:**",
            "- Are there practical limitations that hinder real-world deployment?",
            "- Is the proposed method computationally efficient enough for practical use?",
            "- Does the approach require unrealistic assumptions or resources?",
            "",
            "**Experimental Rigor:**",
            "- Are the experiments comprehensive enough to validate the claims?",
            "- Is there sufficient comparison with strong baselines?",
            "- Are there missing evaluation scenarios (e.g., edge cases, out-of-distribution)?",
            "",
            "**Theoretical Soundness:**",
            "- Are the theoretical assumptions reasonable and well-justified?",
            "- Do the theoretical results directly support the practical claims?",
            "- Are there gaps between theory and implementation?",
            "",
            "## Target Paper",
            f"Title: {target.title}",
            f"Abstract: {target.abstract}",
            fulltext,
            "",
            "## Criteria to Evaluate",
            criteria_list if criteria_list else "(No specific criteria for this theme - use general standards)",
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
            "Remember: Focus on HIGH-LEVEL issues (novelty, practicality, rigor) not just low-level details.",
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
