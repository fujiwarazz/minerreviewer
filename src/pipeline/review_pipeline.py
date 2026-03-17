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
from common.types import (
    ActivatedCriterion,
    ArbiterOutput,
    CalibrationResult,
    Criterion,
    DecisionVerificationReport,
    Paper,
    PaperSignature,
    ScoreConsistencyReport,
    ThemeOutput,
)
from common.utils import read_yaml
from pipeline.aggregate import Aggregator
from pipeline.calibrate import Calibrator
from pipeline.check_score_consistency import ScoreConsistencyChecker
from pipeline.distill_criteria import CriteriaDistiller
from pipeline.distill_experience import DistillationResult, ExperienceDistiller
from pipeline.memory_editor import MemoryEditor
from pipeline.mine_criteria import CriteriaMiner
from pipeline.parse_paper import PaperParser
from pipeline.plan_criteria import CriteriaPlanner
from pipeline.retrieve import Retriever
from pipeline.rewrite_criteria import CriteriaRewriter
from pipeline.verify_decision import DecisionVerifier
from storage.case_store import CaseStore
from storage.doc_store import DocStore
from storage.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class ReviewPipeline:
    """Memory-driven reviewer pipeline"""

    def __init__(self, config_path: str | Path) -> None:
        self.config = read_yaml(config_path)
        self.venue_id = self.config["venue_id"]
        self.doc_store = DocStore()
        self.llm = LLMClient(LLMConfig(**self.config["llm"]))
        self.embedding_client = EmbeddingClient(EmbeddingConfig(**self.config["embedding"]))

        # Initialize new components
        self._init_stores()
        self._init_components()

    def _init_stores(self) -> None:
        """初始化存储组件"""
        memory_cfg = self.config.get("memory", {})
        self.memory_store = MemoryStore(memory_cfg.get("store_path", "data/processed/memory_store.json"))
        self.case_store = CaseStore(
            memory_cfg.get("case_store_path", "data/processed/cases.jsonl"),
            embedding_client=self.embedding_client,
        )

    def _init_components(self) -> None:
        """初始化处理组件"""
        self.paper_parser = PaperParser(self.llm)
        self.criteria_planner = CriteriaPlanner(self.llm)
        self.score_checker = ScoreConsistencyChecker(
            rating_tolerance=self.config.get("score_consistency", {}).get("rating_tolerance", 1.5),
            deviation_threshold=self.config.get("score_consistency", {}).get("deviation_threshold", 2.0),
        )
        self.verifier = DecisionVerifier(self.llm)
        self.experience_distiller = ExperienceDistiller(self.llm)
        self.memory_editor = MemoryEditor(
            memory_store=self.memory_store,
            case_store=self.case_store,
            short_term_utility_threshold=self.config.get("memory", {}).get("short_term_utility_threshold", 0.3),
            long_term_utility_threshold=self.config.get("memory", {}).get("long_term_utility_threshold", 0.6),
        )

    def review_paper(self, paper_id: str, target_year: int | None = None) -> ArbiterOutput:
        papers = self.doc_store.load_papers(self.venue_id)
        target = next((paper for paper in papers if paper.paper_id == paper_id), None)
        if target is None:
            raise ValueError("Paper not found")
        return self._run_review(target, target_year or target.year)

    def _run_review(self, target: Paper, target_year: int | None) -> ArbiterOutput:
        """主审稿流程"""
        retrieval_cfg = self.config["retrieval"]
        distill_cfg = self.config["distill"]
        memory_cfg = self.config["memory"]

        # 1. Parse paper signature
        signature = self._parse_paper(target)

        # 2. Multi-channel retrieval
        bundle = self._retrieve_multi_channel(target, signature, target_year)

        # 3. Mine and distill criteria
        content_criteria, policy_criteria = self._mine_criteria(target, bundle, target_year)

        # 4. Plan criteria with memory
        activated = self._plan_criteria(signature, bundle, content_criteria)

        # 5. Rewrite criteria
        criteria = self._rewrite_criteria(target, activated)

        # 6. Run theme agents
        theme_outputs = self._run_theme_agents(target, criteria)

        # 7. Aggregate with arbiter
        arbiter_output = self._aggregate(theme_outputs, bundle.policy_cards)

        # 8. Verify decision
        verification = self._verify_decision(arbiter_output, target, bundle)

        # 9. Check score consistency (只警告，不改分)
        consistency = self._check_score_consistency(arbiter_output, bundle)

        # 10. Calibrate (多路校准)
        calibration = self._calibrate_multiclass(arbiter_output.raw_rating, target_year)

        # 11. Distill experience
        experience = self._distill_experience(arbiter_output, target, signature, bundle)

        # 12. Update memory
        memory_updates = self._update_memory(experience)

        # 13. Build trace
        arbiter_output.raw_decision = arbiter_output.decision_recommendation
        arbiter_output.trace.update({
            "paper_signature": signature.model_dump() if signature else {},
            "retrieval": bundle.trace,
            "verification": verification.model_dump(),
            "consistency": consistency.model_dump(),
            "calibration": calibration.model_dump(),
            "memory_updates": memory_updates,
            "activated_criteria": [c.model_dump() for c in activated[:10]],
        })

        return arbiter_output

    def _parse_paper(self, paper: Paper) -> PaperSignature:
        """解析论文结构化特征"""
        try:
            return self.paper_parser.parse(paper)
        except Exception as e:
            logger.warning("Failed to parse paper: %s", e)
            return PaperSignature()

    def _retrieve_multi_channel(self, target: Paper, signature: PaperSignature | None, target_year: int | None):
        """多通道检索"""
        retrieval_cfg = self.config["retrieval"]
        retriever = Retriever(
            self.venue_id,
            EmbeddingConfig(**self.config["embedding"]),
            self.config.get("vector_store"),
            case_store=self.case_store,
            memory_store=self.memory_store,
        )
        use_case_memory = self.config.get("retrieval", {}).get("use_case_memory", True)
        bundle = retriever.retrieve(
            target,
            retrieval_cfg["top_k_papers"],
            retrieval_cfg["top_k_reviews"],
            retrieval_cfg.get("unrelated_k", 0),
            retrieval_cfg["similarity_threshold"],
            target_year,
            paper_signature=signature,
            use_case_memory=use_case_memory,
        )
        logger.info(
            "Multi-channel retrieval: %d papers, %d reviews, %d cases, %d policy cards",
            len(bundle.related_papers),
            len(bundle.related_reviews),
            len(bundle.similar_paper_cases),
            len(bundle.policy_cards),
        )
        return bundle

    def _mine_criteria(self, target: Paper, bundle, target_year: int | None):
        """挖掘和精炼标准"""
        distill_cfg = self.config["distill"]

        # Sample reviews for policy mining
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

        miner = CriteriaMiner(self.llm, self.embedding_client, self.config.get("vector_store"))
        content_criteria = miner.mine_content_criteria(target, bundle.related_papers, bundle.related_reviews)
        policy_criteria = miner.mine_policy_criteria(bundle.venue_policy, random_reviews)

        # Distill criteria
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

        logger.info("Criteria: content=%d policy=%d", len(content_criteria), len(policy_criteria))
        return content_criteria, policy_criteria

    def _plan_criteria(
        self,
        signature: PaperSignature | None,
        bundle,
        content_criteria: list[Criterion],
    ) -> list[ActivatedCriterion]:
        """规划激活的标准"""
        max_criteria = self.config.get("distill", {}).get("max_total", 15)
        return self.criteria_planner.plan(
            signature=signature,
            bundle=bundle,
            mined_criteria=content_criteria,
            max_criteria=max_criteria,
        )

    def _rewrite_criteria(self, target: Paper, activated: list[ActivatedCriterion]) -> list[Criterion]:
        """重写标准"""
        criteria = self.criteria_planner.to_criterion_list(activated)
        rewriter = CriteriaRewriter(self.llm)
        return rewriter.rewrite(target, criteria)

    def _aggregate(self, theme_outputs: list[ThemeOutput], policy_cards) -> ArbiterOutput:
        """聚合主题输出"""
        arbiter = ArbiterAgent(AgentConfig(name="arbiter", llm=self.llm))
        aggregator = Aggregator(arbiter)
        # Convert policy cards to criteria for compatibility
        policy_criteria = []
        return aggregator.aggregate(theme_outputs, policy_criteria, None)

    def _verify_decision(
        self,
        arbiter_output: ArbiterOutput,
        paper: Paper,
        bundle,
    ) -> DecisionVerificationReport:
        """验证决策"""
        return self.verifier.verify(arbiter_output, paper, bundle)

    def _check_score_consistency(
        self,
        arbiter_output: ArbiterOutput,
        bundle,
    ) -> ScoreConsistencyReport:
        """检查评分一致性"""
        return self.score_checker.check(arbiter_output, bundle)

    def _calibrate_multiclass(self, raw_rating: float, target_year: int | None) -> CalibrationResult:
        """多路校准"""
        calibration_cfg = self.config.get("calibration", {})
        mode = calibration_cfg.get("mode", "ordinal")

        if calibration_cfg.get("method") in (None, "none"):
            return CalibrationResult(
                calibrated_rating=raw_rating,
                acceptance_likelihood=0.5,
                method="none",
            )

        calibrator = Calibrator(self.venue_id, mode=mode)
        try:
            return calibrator.calibrate(raw_rating)
        except FileNotFoundError:
            # Fit calibrator
            reviews = self.doc_store.load_reviews(self.venue_id)
            if target_year is not None:
                reviews = [r for r in reviews if r.year is None or r.year < target_year]
            calibrator.fit(reviews)
            return calibrator.calibrate(raw_rating)
        except Exception as e:
            logger.warning("Calibration failed: %s", e)
            return CalibrationResult(
                calibrated_rating=raw_rating,
                acceptance_likelihood=0.5,
                method="none",
            )

    def _distill_experience(
        self,
        arbiter_output: ArbiterOutput,
        paper: Paper,
        signature: PaperSignature | None,
        bundle,
    ) -> DistillationResult:
        """蒸馏经验"""
        return self.experience_distiller.distill(
            arbiter_output=arbiter_output,
            paper=paper,
            signature=signature,
            bundle=bundle,
        )

    def _update_memory(self, experience: DistillationResult) -> dict:
        """更新记忆"""
        updates: dict[str, list[str]] = {
            "paper_cases": [],
            "policy_cards": [],
            "critique_cards": [],
            "failure_cards": [],
        }

        # Admit paper case
        if experience.paper_case:
            if self.memory_editor.admit_paper_case(experience.paper_case):
                updates["paper_cases"].append(experience.paper_case.case_id)

        # Admit experience cards
        for card in experience.all_cards():
            result = self.memory_editor.admit(card)
            if result == "admitted_long" or result == "admitted_short":
                if card.kind == "policy":
                    updates["policy_cards"].append(card.card_id)
                elif card.kind == "critique":
                    updates["critique_cards"].append(card.card_id)
                elif card.kind == "failure":
                    updates["failure_cards"].append(card.card_id)

        return updates

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
        max_workers = min(len(themes), 6)

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
            for theme in themes:
                outputs.append(review_theme(theme))

        return outputs
