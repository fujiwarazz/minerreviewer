from __future__ import annotations

import json
import logging
import uuid

from clients.embedding_client import EmbeddingClient
from clients.llm_client import LLMClient
from common.types import Criterion, Paper, Review, VenuePolicy
from storage.milvus_store import MilvusConfig, MilvusStore

logger = logging.getLogger(__name__)


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
        paper_snippets = [f"- {paper.title}: {paper.abstract[:500]}" for paper in related_papers]
        review_snippets = [f"- {review.text}" for review in related_reviews]
        return "\n".join(
            [
                "You are mining content-evaluation criteria for an ICLR review.",
                "Write criteria as concise review questions/checks (not summaries).",
                "Format each criterion as an evaluative question or if/then rule, e.g.,",
                "\"If the authors claim X, do the experiments provide Y evidence?\"",
                "Avoid paper-specific claims; focus on verifiable standards.",
                f"Target paper: {target.title}\n{target.abstract}",
                "Related papers:",
                *paper_snippets,
                "Related reviews:",
                *review_snippets,
                "Return JSON with key 'criteria': list of {text, theme, source_ids}.",
            ]
        )

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
        review_snippets = [f"- {review.text[:240]}" for review in random_reviews]
        return "\n".join(
            [
                "You are mining review style criteria for ICLR-style reviews.",
                "Only output review-writing norms: structure, tone, evidence style, clarity, and meeting-oriented expectations.",
                "Do NOT mention specific paper topics, methods, datasets, or claims.",
                "Few-shot examples (style-only, abstracted):",
                "1) \"If a weakness is raised, it should be tied to a concrete missing experiment or unclear claim.\"",
                "2) \"Summaries should be brief and factual before listing strengths/weaknesses.\"",
                "3) \"Claims should be supported by explicit evidence or citations from the paper.\"",
                "4) \"Avoid speculative statements; state uncertainty when evidence is limited.\"",
                f"Venue policy: {policy_text}",
                "Random review excerpts:",
                *review_snippets,
                "Return JSON with key 'criteria': list of {text, theme, source_ids} focusing on style and structure only.",
            ]
        )

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
