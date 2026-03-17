from __future__ import annotations

import json
import logging
import uuid

from clients.embedding_client import EmbeddingClient
from clients.llm_client import LLMClient
from common.types import Criterion, Paper, Review, VenuePolicy
from storage.milvus_store import MilvusConfig, MilvusStore

logger = logging.getLogger(__name__)

# Standard ICLR review criteria themes
ICLR_CRITERIA_THEMES = {
    "Quality": "Technical soundness, correctness of methods and claims",
    "Clarity": "Paper writing quality, organization, and presentation",
    "Originality": "Novelty of ideas, methods, or insights",
    "Significance": "Importance and impact of contributions",
    "Reproducibility": "Sufficiency of details to reproduce results",
    "Related Work": "Adequate coverage and comparison to prior work",
    "Experiments": "Quality and comprehensiveness of empirical evaluation",
    "Ethics": "Ethical considerations and potential negative impact",
}


class CriteriaMiner:
    def __init__(
        self,
        llm: LLMClient,
        embedding_client: EmbeddingClient | None = None,
        vector_store: dict | None = None,
    ) -> None:
        self.llm = llm
        self.embedding_client = embedding_client
        self.vector_store = vector_store or {}

    def mine_content_criteria(self, target: Paper, related_papers: list[Paper], related_reviews: list[Review]) -> list[Criterion]:
        gap_context = self._coverage_gap_context(target)
        prompt = self._content_prompt(target, related_papers, related_reviews, gap_context)
        response = self.llm.generate_json(prompt)
        items = response.get("criteria", [])
        criteria = self._parse_criteria(items, kind="content")
        logger.info("Mined %s content criteria", len(criteria))
        return criteria

    def mine_policy_criteria(self, venue_policy: VenuePolicy | None, random_reviews: list[Review]) -> list[Criterion]:
        prompt = self._policy_prompt(venue_policy, random_reviews)
        response = self.llm.generate_json(prompt)
        items = response.get("criteria", [])
        criteria = self._parse_criteria(items, kind="policy")
        criteria = self._filter_policy(criteria)
        logger.info("Mined %s policy criteria", len(criteria))
        return criteria

    def _parse_criteria(self, items: list[dict], kind: str) -> list[Criterion]:
        criteria: list[Criterion] = []
        for item in items:
            try:
                source_ids = item.get("source_ids", [])
                if isinstance(source_ids, list):
                    source_ids = [str(value) for value in source_ids]
                elif source_ids is None:
                    source_ids = []
                else:
                    source_ids = [str(source_ids)]
                criteria.append(
                    Criterion(
                        criterion_id=item.get("id") or str(uuid.uuid4()),
                        text=item["text"],
                        theme=item.get("theme", "general"),
                        kind=kind,
                        source_ids=source_ids,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping invalid criterion: %s", exc)
        return criteria

    @staticmethod
    def _content_prompt(target: Paper, related_papers: list[Paper], related_reviews: list[Review], gap_context: str) -> str:
        paper_snippets = [f"- {paper.title}: {paper.abstract[:500]}" for paper in related_papers[:3]]
        review_snippets = [f"- {review.text[:400]}" for review in related_reviews[:6]]

        # Build standard criteria reference
        criteria_ref = "\n".join([f"- {k}: {v}" for k, v in ICLR_CRITERIA_THEMES.items()])

        return "\n".join([
            "You are an expert ICLR reviewer extracting EVALUATION CRITERIA from related work.",
            "",
            "## Standard ICLR Review Themes",
            criteria_ref,
            "",
            "## Your Task",
            "Generate 5-8 specific, verifiable criteria that a reviewer should check for this paper.",
            "Each criterion should be:",
            "1. A concrete question or checklist item (NOT a summary of the paper)",
            "2. Verifiable from the paper content (NOT subjective preference)",
            "3. General enough to apply to similar papers (NOT paper-specific details)",
            "",
            "## Good Criteria Examples:",
            "- \"Does the paper provide a clear problem formulation with formal notation?\"",
            "- \"Are the baseline methods fairly compared with identical evaluation protocols?\"",
            "- \"Does the method section include algorithmic complexity analysis?\"",
            "- \"Are there ablation studies isolating each proposed component's contribution?\"",
            "",
            "## Bad Criteria Examples (AVOID):",
            "- \"The paper proposes X method\" (this is a summary, not a criterion)",
            "- \"Does the paper use dataset Y?\" (too specific to this paper)",
            "- \"Is the writing good?\" (too vague, not verifiable)",
            "",
            "## Target Paper to Review",
            f"Title: {target.title}",
            f"Abstract: {target.abstract}",
            "",
            "## Related Papers (for context)",
            *paper_snippets,
            "",
            "## Related Reviews (for patterns)",
            *review_snippets,
            "",
            f"{gap_context}",
            "",
            "## Output Format",
            "Return JSON with key 'criteria': list of objects with:",
            "- text: the criterion as a question or checklist item",
            "- theme: one of Quality, Clarity, Originality, Significance, Reproducibility, Related Work, Experiments, Ethics",
            "- source_ids: list of review indices that inspired this criterion (e.g., ['review_1', 'review_3'])",
        ])

    def _coverage_gap_context(self, target: Paper) -> str:
        if self.vector_store.get("backend") != "milvus" or self.embedding_client is None:
            return ""
        collection = self.vector_store.get("coverage_gaps_collection")
        if not collection:
            return ""
        milvus_cfg = MilvusConfig(
            host=self.vector_store.get("host", "localhost"),
            port=int(self.vector_store.get("port", 19530)),
            papers_collection=self.vector_store.get("papers_collection", "papers_iclr"),
            reviews_collection=self.vector_store.get("reviews_collection", "reviews_iclr"),
        )
        milvus = MilvusStore(milvus_cfg)
        query = f"{target.title}\n{target.abstract}"
        query_vec = self.embedding_client.embed([query])[0].tolist()
        gap_texts = milvus.search_texts(collection, query_vec, top_k=5)
        if not gap_texts:
            return ""
        return "Common missed points from prior reviews: " + " | ".join(gap_texts)

    @staticmethod
    def _policy_prompt(venue_policy: VenuePolicy | None, random_reviews: list[Review]) -> str:
        policy_text = json.dumps(venue_policy.model_dump() if venue_policy else {}, ensure_ascii=True)
        review_snippets = [f"- {review.text[:300]}" for review in random_reviews[:6]]

        return "\n".join([
            "You are extracting REVIEW STYLE GUIDELINES from ICLR reviews.",
            "",
            "## Your Task",
            "Extract 4-6 style/format guidelines that define what makes a good ICLR review.",
            "These should be about HOW to write reviews, NOT what specific content to mention.",
            "",
            "## Good Policy Examples:",
            "- \"Each weakness should cite a specific section/equation/experiment from the paper\"",
            "- \"Strengths should be stated before weaknesses in the review\"",
            "- \"Avoid absolute language; use 'may' or 'appears to' for uncertain claims\"",
            "- \"Include specific suggestions for improvement, not just criticism\"",
            "- \"Rating should align with the severity of weaknesses identified\"",
            "",
            "## Bad Policy Examples (AVOID):",
            "- \"Mention the dataset used\" (this is content-specific, not style)",
            "- \"Check if the method is novel\" (this is a content criterion, not style)",
            "- \"The paper should be accepted\" (this is a decision, not a guideline)",
            "",
            "## Venue Policy",
            f"{policy_text}",
            "",
            "## Sample Reviews",
            *review_snippets,
            "",
            "## Output Format",
            "Return JSON with key 'criteria': list of objects with:",
            "- text: the style guideline",
            "- theme: 'Style', 'Structure', 'Tone', or 'Evidence'",
            "- source_ids: list of review indices",
        ])

    def _filter_policy(self, criteria: list[Criterion]) -> list[Criterion]:
        if not criteria:
            return []
        prompt = "\n".join(
            [
                "You are a strict filter. Keep only review style/format criteria.",
                "Reject anything about specific paper content, methods, datasets, or results.",
                "Return JSON with key 'keep_ids': list of criterion_id to keep.",
                f"Criteria: {[c.model_dump() for c in criteria]}",
            ]
        )
        try:
            response = self.llm.generate_json(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Policy filter failed: %s", exc)
            return criteria
        keep_ids = response.get("keep_ids")
        if not isinstance(keep_ids, list) or not keep_ids:
            return criteria
        keep_set = {str(item) for item in keep_ids}
        filtered = [c for c in criteria if c.criterion_id in keep_set]
        return filtered if filtered else criteria
