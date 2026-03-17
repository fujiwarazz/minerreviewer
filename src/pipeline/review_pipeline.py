from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agents.arbiter_agent import ArbiterAgent
from agents.base import AgentConfig
from agents.theme_agent import ThemeAgent
from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from clients.llm_client import LLMClient, LLMConfig
from common.types import ArbiterOutput, Criterion, Paper, ThemeOutput
from common.utils import read_yaml
from pipeline.aggregate import Aggregator
from pipeline.calibrate import Calibrator
from pipeline.distill_criteria import CriteriaDistiller
from pipeline.mine_criteria import CriteriaMiner
from pipeline.retrieve import Retriever
from pipeline.rewrite_criteria import CriteriaRewriter
from pipeline.update_memory import update_memory
from storage.doc_store import DocStore
from storage.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class ReviewPipeline:
    def __init__(self, config_path: str | Path) -> None:
        self.config = read_yaml(config_path)
        self.venue_id = self.config["venue_id"]
        self.doc_store = DocStore()
        self.llm = LLMClient(LLMConfig(**self.config["llm"]))
        self.embedding_client = EmbeddingClient(EmbeddingConfig(**self.config["embedding"]))

    def review_paper(self, paper_id: str, target_year: int | None = None) -> ArbiterOutput:
        papers = self.doc_store.load_papers(self.venue_id)
        target = next((paper for paper in papers if paper.paper_id == paper_id), None)
        if target is None:
            raise ValueError("Paper not found")
        return self._run_review(target, target_year or target.year)

    def _run_review(self, target: Paper, target_year: int | None) -> ArbiterOutput:
        retrieval_cfg = self.config["retrieval"]
        distill_cfg = self.config["distill"]
        memory_cfg = self.config["memory"]

        retriever = Retriever(
            self.venue_id,
            EmbeddingConfig(**self.config["embedding"]),
            self.config.get("vector_store"),
        )
        bundle = retriever.retrieve(
            target,
            retrieval_cfg["top_k_papers"],
            retrieval_cfg["top_k_reviews"],
            retrieval_cfg["unrelated_k"],
            retrieval_cfg["similarity_threshold"],
            target_year,
        )
        logger.info(
            "Retrieved %s related papers, %s related reviews, %s unrelated papers",
            len(bundle.related_papers),
            len(bundle.related_reviews),
            len(bundle.unrelated_papers),
        )

        reviews_pool = [review for review in self.doc_store.load_reviews(self.venue_id) if review.paper_id != target.paper_id]
        if target_year is not None:
            reviews_pool = [review for review in reviews_pool if review.year is None or review.year < target_year]
        policy_cfg = self.config.get("policy_mining", {})
        accept_count = int(policy_cfg.get("sample_accept", 3))
        reject_count = int(policy_cfg.get("sample_reject", 3))
        accept_reviews = [r for r in reviews_pool if r.decision and str(r.decision).lower().startswith("accept")]
        reject_reviews = [r for r in reviews_pool if r.decision and str(r.decision).lower().startswith("reject")]
        sampled_accept = random.sample(accept_reviews, min(accept_count, len(accept_reviews)))
        sampled_reject = random.sample(reject_reviews, min(reject_count, len(reject_reviews)))
        random_reviews = sampled_accept + sampled_reject
        logger.info("Sampled %s accept/%s reject reviews for policy mining", len(sampled_accept), len(sampled_reject))

        miner = CriteriaMiner(self.llm, self.embedding_client, self.config.get("vector_store"))
        content_criteria = miner.mine_content_criteria(target, bundle.related_papers, bundle.related_reviews)
        policy_criteria = miner.mine_policy_criteria(bundle.venue_policy, random_reviews)
        logger.info("Criteria mined: content=%s policy=%s", len(content_criteria), len(policy_criteria))

        distiller = CriteriaDistiller(
            self.config["embedding"]["model"],
            embedder=lambda texts, _: self.embedding_client.embed(texts),
        )
        content_criteria = distiller.dedup(content_criteria, distill_cfg["dedup_threshold"])
        content_criteria = distiller.select(
            content_criteria,
            distill_cfg["max_total"],
            distill_cfg["max_per_theme"],
            distill_cfg["seed"],
            distill_cfg.get("strategy"),
            distill_cfg.get("epsilon", 1.0),
        )

        policy_criteria = distiller.dedup(policy_criteria, distill_cfg["dedup_threshold"])
        policy_criteria = distiller.select(
            policy_criteria,
            distill_cfg["max_total"],
            distill_cfg["max_per_theme"],
            distill_cfg["seed"],
            distill_cfg.get("strategy"),
            distill_cfg.get("epsilon", 1.0),
        )
        logger.info("Criteria selected: content=%s policy=%s", len(content_criteria), len(policy_criteria))

        rewriter = CriteriaRewriter(self.llm)
        content_criteria = rewriter.rewrite(target, content_criteria)
        logger.info("Criteria rewritten: content=%s", len(content_criteria))

        theme_outputs = self._run_theme_agents(target, content_criteria)
        logger.info("Theme agent outputs: %s", len(theme_outputs))
        arbiter = ArbiterAgent(AgentConfig(name="arbiter", llm=self.llm))
        aggregator = Aggregator(arbiter)
        arbiter_output = aggregator.aggregate(theme_outputs, policy_criteria, bundle.venue_policy)

        decision_cfg = self.config.get("decision_scoring", {})
        if decision_cfg.get("use_similarity", True):
            arbiter_output = self._score_with_similar_reviews(
                target,
                arbiter_output,
                decision_cfg,
                target_year,
            )
        arbiter_output.raw_decision = arbiter_output.decision_recommendation

        acceptance = None
        if self.config.get("calibration", {}).get("method") not in (None, "none"):
            calibrator = Calibrator(self.venue_id)
            try:
                acceptance = calibrator.predict(arbiter_output.raw_rating)
            except Exception:  # noqa: BLE001
                reviews = self.doc_store.load_reviews(self.venue_id)
                if target_year is not None:
                    reviews = [review for review in reviews if review.year is None or review.year < target_year]
                artifact = calibrator.fit(reviews)
                if artifact is not None:
                    acceptance = calibrator.predict(arbiter_output.raw_rating)
                else:
                    logger.info("Calibrator not available; skipping")
        if acceptance is not None:
            arbiter_output.acceptance_likelihood = acceptance
            arbiter_output.calibrated_rating = acceptance

        store = MemoryStore(memory_cfg["store_path"])
        updated_cards = update_memory(
            store,
            self.venue_id,
            policy_criteria,
            arbiter_output.raw_rating,
            arbiter_output.acceptance_likelihood or arbiter_output.raw_rating,
            memory_cfg["similarity_threshold"],
            memory_cfg["stable_margin"],
            memory_cfg["borderline_low"],
            memory_cfg["borderline_high"],
            trace=bundle.trace,
        )

        arbiter_output.trace.update(
            {
                "retrieval": bundle.trace,
                "criteria": {
                    "content": [c.criterion_id for c in content_criteria],
                    "policy": [c.criterion_id for c in policy_criteria],
                },
                "criteria_details": {
                    "content": [c.model_dump() for c in content_criteria],
                    "policy": [c.model_dump() for c in policy_criteria],
                },
                "decision_scoring": self.config.get("decision_scoring", {}),
                "memory_updates": updated_cards,
                "calibration": self.config["calibration"]["method"] if acceptance is not None else "none",
            }
        )
        return arbiter_output

    def _run_theme_agents(self, target: Paper, criteria: list[Criterion]) -> list[ThemeOutput]:
        themes = list(self.config.get("themes", []))
        criteria_themes = [c.theme for c in criteria if c.theme]
        for theme in criteria_themes:
            if theme not in themes:
                themes.append(theme)
        review_cfg = self.config.get("review", {})
        use_fulltext = bool(review_cfg.get("use_fulltext", False))
        max_fulltext_chars = int(review_cfg.get("max_fulltext_chars", 12000))

        # Parallel execution of theme agents
        outputs: list[ThemeOutput] = []
        max_workers = min(len(themes), 6)  # Limit concurrent threads

        def review_theme(theme: str) -> ThemeOutput:
            themed = [c for c in criteria if c.theme == theme]
            agent = ThemeAgent(
                AgentConfig(name=f"theme_{theme}", llm=self.llm),
                theme,
                use_fulltext=use_fulltext,
                max_fulltext_chars=max_fulltext_chars,
            )
            return agent.review(target, themed)

        if max_workers > 1 and len(themes) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(review_theme, theme): theme for theme in themes}
                for future in as_completed(futures):
                    try:
                        outputs.append(future.result())
                    except Exception as e:
                        theme = futures[future]
                        logger.error("Theme agent %s failed: %s", theme, e)
        else:
            # Fallback to sequential for single theme or safety
            for theme in themes:
                outputs.append(review_theme(theme))

        return outputs

    @staticmethod
    def _decision_from_acceptance(acceptance: float) -> str:
        if acceptance >= 0.6:
            return "accept"
        if acceptance <= 0.4:
            return "reject"
        return "borderline"

    def _score_with_similar_reviews(
        self,
        target: Paper,
        arbiter_output: ArbiterOutput,
        decision_cfg: dict,
        target_year: int | None,
    ) -> ArbiterOutput:
        top_k = int(decision_cfg.get("top_k", 8))
        retriever = Retriever(
            self.venue_id,
            EmbeddingConfig(**self.config["embedding"]),
            self.config.get("vector_store"),
        )
        pseudo_review = self._format_pseudo_review(arbiter_output.strengths, arbiter_output.weaknesses)
        similar_reviews = retriever.retrieve_similar_reviews(pseudo_review, top_k, target_year, target.paper_id)
        if not similar_reviews:
            return arbiter_output
        prompt = self._decision_prompt(target, pseudo_review, similar_reviews)
        response = self.llm.generate_json(prompt)
        raw_rating = response.get("raw_rating", arbiter_output.raw_rating)
        decision = response.get("decision_recommendation", arbiter_output.decision_recommendation)
        arbiter_output.raw_rating = float(raw_rating) if raw_rating is not None else arbiter_output.raw_rating
        arbiter_output.decision_recommendation = decision
        arbiter_output.trace["similar_reviews_used"] = [
            {
                "review_id": r.review_id,
                "paper_id": r.paper_id,
                "rating": r.rating,
                "decision": r.decision,
            }
            for r in similar_reviews
        ]
        return arbiter_output

    @staticmethod
    def _format_pseudo_review(strengths: list[str], weaknesses: list[str]) -> str:
        parts = ["Strengths:"]
        parts.extend([f"- {item}" for item in strengths])
        parts.append("Weaknesses:")
        parts.extend([f"- {item}" for item in weaknesses])
        return "\n".join(parts)

    @staticmethod
    def _decision_prompt(target: Paper, pseudo_review: str, similar_reviews: list[Review]) -> str:
        examples = [
            {
                "review": review.text[:800],
                "rating": review.rating,
                "decision": review.decision,
            }
            for review in similar_reviews
        ]
        return "\n".join(
            [
                "You are assigning a rating and decision based on similar historical reviews.",
                f"Target paper: {target.title}\n{target.abstract}",
                f"Generated review:\n{pseudo_review}",
                f"Similar reviews with ratings/decisions: {examples}",
                "Return JSON with keys raw_rating (float) and decision_recommendation (accept/reject/borderline/revise).",
            ]
        )
