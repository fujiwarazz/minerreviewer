from __future__ import annotations

import logging
import random
from collections import Counter
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
from storage.deepreview_store import DeepReviewCaseStore
from storage.doc_store import DocStore
from storage.memory_store import MemoryStore
from storage.memory_registry import MemoryRegistry
from storage.multi_case_store import MultiCaseStore
from storage.multi_memory_store import MultiMemoryStore
from storage.multi_vector_memory_store import MultiVectorMemoryStore
from storage.milvus_store import MilvusConfig
from pipeline.agent_memory_allocator import AgentMemoryAllocator

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
        """初始化存储组件 - 支持热插拔记忆库"""
        memory_cfg = self.config.get("memory", {})
        retrieval_cfg = self.config.get("retrieval", {})

        # 使用 MemoryRegistry 管理热插拔记忆库
        registry_path = memory_cfg.get("registry_path", "data/processed/registry.json")
        self.registry = MemoryRegistry(registry_path)

        # 使用 MultiCaseStore 和 MultiMemoryStore
        use_hot_swap = memory_cfg.get("use_hot_swap", True)

        if use_hot_swap:
            # 热插拔模式：从注册表加载活跃记忆库
            self.case_store = MultiCaseStore(
                registry=self.registry,
                embedding_client=self.embedding_client,
                milvus_config=MilvusConfig(
                    host=self.config.get("vector_store", {}).get("host", "localhost"),
                    port=int(self.config.get("vector_store", {}).get("port", 19530)),
                    papers_collection="",
                    reviews_collection="",
                ) if self.config.get("vector_store", {}).get("backend") == "milvus" else None,
                embedding_weight=retrieval_cfg.get("case_embedding_weight", 0.5),
                signature_weight=retrieval_cfg.get("signature_weight", 0.4),
                venue_match_bonus=retrieval_cfg.get("venue_match_bonus", 0.1),
            )
            self.memory_store = MultiMemoryStore(registry=self.registry)
            logger.info("Using hot-swap memory: %d active memories", len(self.registry.get_active_memories()))
        else:
            # 传统模式：单一文件路径
            self.memory_store = MemoryStore(memory_cfg.get("store_path", "data/processed/memory_store.json"))
            self.case_store = CaseStore(
                memory_cfg.get("case_store_path", "data/processed/cases.jsonl"),
                embedding_client=self.embedding_client,
                milvus_config=MilvusConfig(
                    host=self.config.get("vector_store", {}).get("host", "localhost"),
                    port=int(self.config.get("vector_store", {}).get("port", 19530)),
                    papers_collection="",
                    reviews_collection="",
                ) if self.config.get("vector_store", {}).get("backend") == "milvus" else None,
            )

        # 初始化 DeepReview 记忆库（热插拔）
        self.deepreview_store: DeepReviewCaseStore | None = None
        deepreview_cfg = memory_cfg.get("deepreview", {})
        if deepreview_cfg.get("enabled", False):
            try:
                deepreview_path = deepreview_cfg.get("path", "data/processed/deepreview_cases_full.jsonl")
                self.deepreview_store = DeepReviewCaseStore(
                    path=deepreview_path,
                    embedding_client=self.embedding_client,
                    milvus_config=MilvusConfig(
                        host=self.config.get("vector_store", {}).get("host", "localhost"),
                        port=int(self.config.get("vector_store", {}).get("port", 19530)),
                        papers_collection="",
                        reviews_collection="",
                    ) if self.config.get("vector_store", {}).get("backend") == "milvus" else None,
                    primary_area_weight=deepreview_cfg.get("primary_area_weight", 0.1),
                )
                logger.info(
                    "Loaded DeepReview memory: %d cases, %d areas",
                    len(self.deepreview_store.cases),
                    len(self.deepreview_store.list_areas()),
                )
            except Exception as e:
                logger.warning("Failed to load DeepReview memory: %s", e)

        # === 新增：VectorMemoryStore初始化 ===
        vector_memory_cfg = memory_cfg.get("vector_memory", {})
        if vector_memory_cfg.get("enabled", True):
            self.vector_memory_store = MultiVectorMemoryStore(
                registry=self.registry,
                embedding_client=self.embedding_client,
                milvus_config=MilvusConfig(
                    host=self.config.get("vector_store", {}).get("host", "localhost"),
                    port=int(self.config.get("vector_store", {}).get("port", 19530)),
                    papers_collection="",
                    reviews_collection="",
                ) if self.config.get("vector_store", {}).get("backend") == "milvus" else None,
            )
            logger.info(
                "Initialized VectorMemoryStore with %d active memories",
                len(self.registry.get_active_memories()),
            )
        else:
            self.vector_memory_store = None

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

        # === 新增：AgentMemoryAllocator ===
        agent_allocation_cfg = self.config.get("memory", {}).get("agent_allocation", {})
        if agent_allocation_cfg.get("enabled", True):
            self.memory_allocator = AgentMemoryAllocator()
            logger.info("Initialized AgentMemoryAllocator")
        else:
            self.memory_allocator = None

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
        content_criteria, policy_criteria = self._mine_criteria(target, signature, bundle, target_year)

        # 4. Plan criteria with memory
        activated = self._plan_criteria(signature, bundle, content_criteria, policy_criteria)

        # 5. Rewrite criteria
        criteria = self._rewrite_criteria(target, activated)

        # 6. Run theme agents (with agent memories)
        theme_outputs = self._run_theme_agents(target, criteria, bundle)

        # 7. Aggregate with arbiter (now includes policy_criteria and venue_policy)
        arbiter_output = self._aggregate(
            theme_outputs, bundle.policy_cards, policy_criteria, bundle.venue_policy,
            similar_cases=bundle.similar_paper_cases,
        )

        # 8. Verify decision
        verification = self._verify_decision(arbiter_output, target, bundle)

        # 9. If verification requires revision, trigger arbiter revise
        if verification.requires_revision:
            arbiter_output = self._revise_decision(
                arbiter_output, target, verification, bundle, theme_outputs
            )
            # Re-verify after revision
            verification = self._verify_decision(arbiter_output, target, bundle)

        # 10. Check score consistency (只警告，不改分)
        consistency = self._check_score_consistency(arbiter_output, bundle)

        # 11. Calibrate (多路校准)
        calibration = self._calibrate_multiclass(arbiter_output.raw_rating, target_year)

        # 12. Apply calibration to final output (回填校准结果)
        arbiter_output = self._apply_calibration(arbiter_output, calibration)

        # 13. Distill experience
        experience = self._distill_experience(arbiter_output, target, signature, bundle)

        # 14. Update memory
        memory_updates = self._update_memory(experience)

        # 15. Build final report with interpretability fields
        arbiter_output.raw_decision = arbiter_output.decision_recommendation

        # Fill in interpretability summaries
        if consistency.warning:
            arbiter_output.consistency_summary = (
                f"Consistency level: {consistency.consistency_level}. "
                f"Based on {consistency.similar_review_count} similar cases. "
                f"{consistency.warning}"
            )
        else:
            arbiter_output.consistency_summary = (
                f"Consistency level: {consistency.consistency_level}. "
                f"Based on {consistency.similar_review_count} similar cases."
            )

        # Extract key decisive issues from weaknesses if not set
        if not arbiter_output.key_decisive_issues:
            arbiter_output.key_decisive_issues = [
                w[:100] + "..." if len(w) > 100 else w
                for w in arbiter_output.weaknesses[:3]
            ]

        # Generate decision rationale if not set
        if not arbiter_output.decision_rationale:
            decision = arbiter_output.decision_recommendation or "borderline"
            rating = arbiter_output.raw_rating
            arbiter_output.decision_rationale = (
                f"Decision: {decision} (rating: {rating:.1f}). "
                f"Key factors: {len(arbiter_output.strengths)} strengths, "
                f"{len(arbiter_output.weaknesses)} weaknesses identified."
            )

        arbiter_output.trace.update({
            "paper_signature": signature.model_dump() if signature else {},
            "retrieval": bundle.trace,
            "memory_channel_usage": self._build_memory_channel_usage(
                bundle=bundle,
                activated=activated,
                rewritten_criteria=criteria,
                theme_outputs=theme_outputs,
                arbiter_output=arbiter_output,
            ),
            "verification": verification.model_dump(),
            "consistency": consistency.model_dump(),
            "calibration": calibration.model_dump(),
            "memory_updates": memory_updates,
            "activated_criteria": [c.model_dump() for c in activated[:10]],
            "final_decision_flow": {
                "initial_rating": arbiter_output.raw_rating,
                "was_revised": verification.requires_revision,
                "calibrated_acceptance": arbiter_output.acceptance_likelihood,
                "consistency_warning": consistency.warning,
            },
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
            deepreview_store=self.deepreview_store,  # 热插拔 DeepReview 记忆
            vector_memory_store=self.vector_memory_store,  # 新增：向量记忆存储
        )
        use_case_memory = self.config.get("retrieval", {}).get("use_case_memory", True)
        use_agent_memory = self.config.get("retrieval", {}).get("use_agent_memory", True)  # 新增

        # 从 signature 获取 primary_area（如果有）
        primary_area = signature.domain if signature else None

        bundle = retriever.retrieve(
            target,
            retrieval_cfg["top_k_papers"],
            retrieval_cfg["top_k_reviews"],
            retrieval_cfg.get("unrelated_k", 0),
            retrieval_cfg["similarity_threshold"],
            target_year,
            paper_signature=signature,
            use_case_memory=use_case_memory,
            use_agent_memory=use_agent_memory,  # 新增
            primary_area=primary_area,  # 新增：领域匹配
        )
        logger.info(
            "Multi-channel retrieval: %d papers, %d reviews, %d cases, %d policy cards, %d agent memories",
            len(bundle.related_papers),
            len(bundle.related_reviews),
            len(bundle.similar_paper_cases),
            len(bundle.policy_cards),
            sum(len(cards) for cards in bundle.agent_memories.values()),
        )
        return bundle

    def _mine_criteria(self, target: Paper, signature, bundle, target_year: int | None):
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

        # 提取领域信息用于 criteria mining
        domain = signature.domain if signature else None

        miner = CriteriaMiner(self.llm, self.embedding_client, self.config.get("vector_store"))
        content_criteria = miner.mine_content_criteria(domain, bundle.related_papers, bundle.related_reviews)
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
        policy_criteria: list[Criterion],
    ) -> list[ActivatedCriterion]:
        """规划激活的标准"""
        max_criteria = self.config.get("distill", {}).get("max_total", 15)
        return self.criteria_planner.plan(
            signature=signature,
            bundle=bundle,
            mined_criteria=content_criteria,
            mined_policy_criteria=policy_criteria,
            max_criteria=max_criteria,
        )

    def _rewrite_criteria(self, target: Paper, activated: list[ActivatedCriterion]) -> list[Criterion]:
        """重写标准"""
        criteria = self.criteria_planner.to_criterion_list(activated)
        rewriter = CriteriaRewriter(self.llm)
        return rewriter.rewrite(target, criteria)

    def _aggregate(
        self,
        theme_outputs: list[ThemeOutput],
        policy_cards: list,
        policy_criteria: list[Criterion],
        venue_policy,
        similar_cases: list[PaperCase] | None = None,
    ) -> ArbiterOutput:
        """
        聚合主题输出

        现在真正使用 policy_criteria 和 venue_policy 进入最终决策
        同时参考相似案例的评分分布来校准打分
        """
        arbiter = ArbiterAgent(AgentConfig(name="arbiter", llm=self.llm))
        aggregator = Aggregator(arbiter)

        # Convert policy cards to criteria format for arbiter
        policy_from_memory = []
        for card in policy_cards:
            memory_prefix = "domain_memory" if getattr(card, "scope", None) == "domain" else "policy_memory"
            policy_from_memory.append(Criterion(
                criterion_id=f"{memory_prefix}_{card.card_id[:8]}",
                text=card.content,
                theme=card.theme,
                kind="policy",
                source_ids=[f"memory:{card.card_id}", f"scope:{getattr(card, 'scope', 'venue')}"],
            ))

        # Merge all policy criteria: memory + mined
        all_policy_criteria = policy_from_memory + policy_criteria

        logger.info(
            "Aggregating with %d theme outputs, %d policy criteria (memory=%d, mined=%d), %d similar cases",
            len(theme_outputs),
            len(all_policy_criteria),
            len(policy_from_memory),
            len(policy_criteria),
            len(similar_cases) if similar_cases else 0,
        )

        return aggregator.aggregate(theme_outputs, all_policy_criteria, venue_policy, similar_cases)

    def _build_memory_channel_usage(
        self,
        bundle,
        activated: list[ActivatedCriterion],
        rewritten_criteria: list[Criterion],
        theme_outputs: list[ThemeOutput],
        arbiter_output: ArbiterOutput,
    ) -> dict:
        """汇总各类 memory channel 的检索/规划/消费统计"""
        criterion_source_by_id = {
            criterion.criterion_id: activated[idx].source
            for idx, criterion in enumerate(rewritten_criteria)
            if idx < len(activated)
        }

        theme_used_ids = [
            criterion_id
            for output in theme_outputs
            for criterion_id in output.criteria_used
        ]
        theme_used_sources = Counter(
            criterion_source_by_id[criterion_id]
            for criterion_id in theme_used_ids
            if criterion_id in criterion_source_by_id
        )

        arbiter_used_ids = arbiter_output.trace.get("criteria_used", [])
        arbiter_used_sources = Counter()
        for criterion_id in arbiter_used_ids:
            if criterion_id.startswith("domain_memory_"):
                arbiter_used_sources["domain_memory"] += 1
            elif criterion_id.startswith("policy_memory_"):
                arbiter_used_sources["policy_memory"] += 1
            else:
                arbiter_used_sources["policy_mined"] += 1

        planned_sources = Counter(item.source for item in activated)
        retrieved_policy_cards = Counter(
            "domain_memory" if getattr(card, "scope", None) == "domain" else "policy_memory"
            for card in bundle.policy_cards
        )

        return {
            "retrieved": {
                "policy_memory": retrieved_policy_cards.get("policy_memory", 0),
                "domain_memory": retrieved_policy_cards.get("domain_memory", 0),
                "case_memory": len(bundle.similar_paper_cases),
                "critique_memory": len(bundle.critique_cases),
                "failure_memory": len(bundle.failure_cards),
            },
            "planned": {
                "policy_memory": planned_sources.get("policy_memory", 0),
                "domain_memory": planned_sources.get("domain_memory", 0),
                "case_memory": planned_sources.get("case_memory", 0),
                "failure_memory": planned_sources.get("failure_memory", 0),
                "policy_mined": planned_sources.get("policy_mined", 0),
                "content_mined": planned_sources.get("content_mined", 0),
            },
            "consumed": {
                "theme": {
                    "policy_memory": theme_used_sources.get("policy_memory", 0),
                    "domain_memory": theme_used_sources.get("domain_memory", 0),
                    "case_memory": theme_used_sources.get("case_memory", 0),
                    "failure_memory": theme_used_sources.get("failure_memory", 0),
                    "policy_mined": theme_used_sources.get("policy_mined", 0),
                    "content_mined": theme_used_sources.get("content_mined", 0),
                },
                "arbiter": {
                    "policy_memory": arbiter_used_sources.get("policy_memory", 0),
                    "domain_memory": arbiter_used_sources.get("domain_memory", 0),
                    "policy_mined": arbiter_used_sources.get("policy_mined", 0),
                    "case_anchor": len(bundle.similar_paper_cases),
                },
            },
        }

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
            target_year=paper.year,
        )

    def _update_memory(self, experience: DistillationResult) -> dict:
        """更新记忆 - 新增向量存储和agent分配"""
        updates: dict[str, list[str]] = {
            "paper_cases": [],
            "policy_cards": [],
            "critique_cards": [],
            "failure_cards": [],
            "agent_memories": [],  # 新增
        }

        # Admit paper case
        if experience.paper_case:
            if self.memory_editor.admit_paper_case(experience.paper_case):
                updates["paper_cases"].append(experience.paper_case.case_id)

        # Admit experience cards
        for card in experience.all_cards():
            result = self.memory_editor.admit(card)
            if result == "admitted_long" or result == "admitted_short":
                if card.kind == "strength":
                    updates["policy_cards"].append(card.card_id)
                elif card.kind == "critique":
                    updates["critique_cards"].append(card.card_id)
                elif card.kind == "failure":
                    updates["failure_cards"].append(card.card_id)

        # === 新增：分配卡片到agent并存储到VectorMemoryStore ===
        if self.vector_memory_store and self.memory_allocator:
            all_cards = experience.all_cards()
            if all_cards:
                agent_allocation_cfg = self.config.get("memory", {}).get("agent_allocation", {})
                share_strength_with_arbiter = agent_allocation_cfg.get("share_strength_with_arbiter", True)

                allocation = self.memory_allocator.allocate(
                    all_cards,
                    share_strength_with_arbiter=share_strength_with_arbiter,
                )

                for agent_name, cards in allocation.items():
                    for card in cards:
                        # 根据卡片年份找到对应的memory库
                        memory_year = card.metadata.get("memory_year", experience.paper_case.year if experience.paper_case else 2024)
                        memory_id = f"{card.venue_id or self.venue_id}_{memory_year}_learned"

                        # 尝试添加到对应store
                        card_ids = self.vector_memory_store.batch_add_cards_to_store(
                            memory_id,
                            [card],
                            owner_agent=agent_name,
                        )
                        if card_ids:
                            updates["agent_memories"].extend(card_ids)

                logger.info(
                    "Allocated and stored %d agent memories to %d stores",
                    len(updates["agent_memories"]),
                    len(self.vector_memory_store._stores),
                )

        return updates

    def _run_theme_agents(self, target: Paper, criteria: list[Criterion], bundle) -> list[ThemeOutput]:
        """运行主题Agent，支持agent个人记忆

        Args:
            target: 目标论文
            criteria: 审稿标准列表
            bundle: RetrievalBundle（包含policy_cards, critique_cases, agent_memories）

        Returns:
            list of ThemeOutput
        """
        # 核心 themes - 始终评估这些维度
        core_themes = ["Clarity", "Quality", "Originality", "Significance", "Experiments"]
        themes = list(self.config.get("themes", []))

        # 添加核心 themes
        for t in core_themes:
            if t not in themes:
                themes.append(t)

        # 添加 criteria 中的 themes
        criteria_themes = [c.theme for c in criteria if c.theme]
        for theme in criteria_themes:
            if theme not in themes:
                themes.append(theme)
        review_cfg = self.config.get("review", {})
        use_fulltext = bool(review_cfg.get("use_fulltext", False))
        max_fulltext_chars = int(review_cfg.get("max_fulltext_chars", 12000))

        # Filter policy cards by theme
        policy_cards = bundle.policy_cards or []
        themed_policies = {}
        for card in policy_cards:
            # Handle both dict and Pydantic model
            if hasattr(card, 'theme'):
                card_theme = card.theme.lower() if card.theme else "general"
            else:
                card_theme = card.get("theme", "general").lower()
            if card_theme not in themed_policies:
                themed_policies[card_theme] = []
            themed_policies[card_theme].append(card)

        # Filter critique cards by theme (criticism patterns for weaknesses)
        critique_cards = bundle.critique_cases or []
        themed_critiques = {}
        for card in critique_cards:
            if hasattr(card, 'theme'):
                card_theme = card.theme.lower() if card.theme else "general"
            else:
                card_theme = card.get("theme", "general").lower()
            if card_theme not in themed_critiques:
                themed_critiques[card_theme] = []
            themed_critiques[card_theme].append(card)

        # === 新增：提取agent个人记忆 ===
        agent_memories = bundle.agent_memories or {}

        # Parallel execution of theme agents
        outputs: list[ThemeOutput] = []
        max_workers = min(len(themes), 6)
        logger.info("Running theme agents for %d themes with max_workers=%d", len(themes), max_workers)

        def review_theme(theme: str) -> ThemeOutput:
            MIN_CRITERIA_PER_THEME = 2  # 每个主题最少标准数
            themed = [c for c in criteria if c.theme == theme]

            # 如果该 theme 标准不足，借用其他 theme 的高优先级标准
            if len(themed) < MIN_CRITERIA_PER_THEME:
                # 获取其他 theme 的标准，按 priority 排序
                others = [c for c in criteria if c.theme != theme]
                others.sort(key=lambda x: x.priority if hasattr(x, 'priority') else 0, reverse=True)
                needed = MIN_CRITERIA_PER_THEME - len(themed)
                themed = themed + others[:needed]
                if others[:needed]:
                    logger.info(
                        "Theme '%s' borrowed %d criteria from other themes (had %d, needed %d)",
                        theme, len(others[:needed]), len([c for c in criteria if c.theme == theme]), MIN_CRITERIA_PER_THEME
                    )

            themed_pol = themed_policies.get(theme.lower(), [])[:10]  # 每个主题最多10条 policy
            themed_crit = themed_critiques.get(theme.lower(), [])[:8]  # 每个主题最多8条 critique

            # === 新增：获取该theme agent的个人记忆 ===
            agent_name = f"theme_{theme.lower()}"
            themed_memories = agent_memories.get(agent_name, [])

            agent = ThemeAgent(
                AgentConfig(name=f"theme_{theme}", llm=self.llm),
                theme,
                use_fulltext=use_fulltext,
                max_fulltext_chars=max_fulltext_chars,
            )
            return agent.review(target, themed, themed_pol, themed_crit, agent_memories=themed_memories)

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

    def _revise_decision(
        self,
        arbiter_output: ArbiterOutput,
        paper: Paper,
        verification: DecisionVerificationReport,
        bundle,
        theme_outputs: list[ThemeOutput],
    ) -> ArbiterOutput:
        """
        当 verification.requires_revision 时，触发 arbiter revise

        这是决策链的关键闭环：verification 诊断会真正影响最终决策
        """
        logger.warning(
            "Decision verification requires revision: %s",
            verification.warnings
        )

        # Build revision context
        revision_prompt = self._build_revision_prompt(
            arbiter_output, paper, verification, bundle
        )

        try:
            response = self.llm.generate_json(revision_prompt)

            # Update arbiter output with revision
            if "raw_rating" in response:
                old_rating = arbiter_output.raw_rating
                arbiter_output.raw_rating = float(response["raw_rating"])
                logger.info("Rating revised: %.1f -> %.1f", old_rating, arbiter_output.raw_rating)

            if "decision_recommendation" in response:
                old_decision = arbiter_output.decision_recommendation
                arbiter_output.decision_recommendation = response["decision_recommendation"]
                logger.info("Decision revised: %s -> %s", old_decision, arbiter_output.decision_recommendation)

            if "revised_strengths" in response:
                arbiter_output.strengths = response["revised_strengths"]

            if "revised_weaknesses" in response:
                arbiter_output.weaknesses = response["revised_weaknesses"]

            # 更新可解释性字段
            if "decision_rationale" in response:
                arbiter_output.decision_rationale = response["decision_rationale"]

            if "key_decisive_issues" in response:
                arbiter_output.key_decisive_issues = response["key_decisive_issues"]

            # Record revision in trace
            arbiter_output.trace["revision"] = {
                "reason": verification.warnings,
                "old_rating": arbiter_output.trace.get("initial_rating"),
                "new_rating": arbiter_output.raw_rating,
                "was_revised": True,
            }

            # 更新验证摘要
            arbiter_output.verification_summary = (
                f"Revision triggered due to: {', '.join(verification.warnings[:2])}. "
                f"Score-text alignment: {verification.score_text_alignment}."
            )

        except Exception as e:
            logger.error("Failed to revise decision: %s", e)
            # Keep original output if revision fails

        return arbiter_output

    def _build_revision_prompt(
        self,
        arbiter_output: ArbiterOutput,
        paper: Paper,
        verification: DecisionVerificationReport,
        bundle,
    ) -> str:
        """构建修订提示"""
        import statistics

        # Build anchor rating section from similar cases
        anchor_section = ""
        similar_cases = bundle.similar_paper_cases
        if similar_cases:
            valid_cases = [c for c in similar_cases if c.rating is not None]
            if valid_cases:
                ratings = [c.rating for c in valid_cases]
                mean_rating = statistics.mean(ratings)
                accept_count = sum(1 for c in valid_cases if c.decision and "accept" in c.decision.lower())
                reject_count = sum(1 for c in valid_cases if c.decision and "reject" in c.decision.lower())

                # Calculate suggested range
                if accept_count > reject_count:
                    suggested_range = f"{max(4.0, mean_rating - 1.0):.1f} - {min(10.0, mean_rating + 1.0):.1f}"
                    suggested_decision = "Accept or Borderline"
                elif reject_count > accept_count:
                    suggested_range = f"{max(1.0, mean_rating - 1.5):.1f} - {min(6.0, mean_rating + 1.0):.1f}"
                    suggested_decision = "Reject or Borderline"
                else:
                    suggested_range = f"{max(2.0, mean_rating - 1.0):.1f} - {min(8.0, mean_rating + 1.0):.1f}"
                    suggested_decision = "Borderline"

                anchor_section = "\n".join([
                    "",
                    "═══════════════════════════════════════════════════════════════",
                    "⚠️ RATING CALIBRATION (CRITICAL FOR REVISION)",
                    "",
                    f"Anchor Rating: {mean_rating:.1f} (mean of {len(valid_cases)} similar papers)",
                    f"Decision Distribution: Accept={accept_count}, Reject={reject_count}",
                    "",
                    f"Your revised rating should be within: {suggested_range}",
                    f"Suggested decision: {suggested_decision}",
                    "",
                    "⚠️ DO NOT deviate more than ±1.5 from the anchor rating!",
                    "═══════════════════════════════════════════════════════════════",
                ])

        return "\n".join([
            "The initial review decision has failed verification. Please revise based on the following issues:",
            "",
            f"Paper: {paper.title}",
            "",
            "Initial Review:",
            f"- Rating: {arbiter_output.raw_rating}",
            f"- Decision: {arbiter_output.decision_recommendation}",
            f"- Strengths: {arbiter_output.strengths[:3]}",
            f"- Weaknesses: {arbiter_output.weaknesses[:3]}",
            "",
            "Verification Issues:",
            *[f"- {w}" for w in verification.warnings],
            "",
            f"Score-Text Alignment: {verification.score_text_alignment}",
            f"Evidence Support: {verification.evidence_support_level}",
            anchor_section,
            "",
            "Please revise and return JSON with:",
            "- raw_rating (float): revised rating (must respect anchor rating bounds)",
            "- decision_recommendation (string): accept/reject/borderline",
            "- revised_strengths (list, optional): updated strengths",
            "- revised_weaknesses (list, optional): updated weaknesses",
            "- decision_rationale (string, optional): brief explanation of the decision",
            "- key_decisive_issues (list, optional): the 1-3 most decisive issues that determined the outcome",
        ])

    def _apply_calibration(
        self,
        arbiter_output: ArbiterOutput,
        calibration: CalibrationResult,
    ) -> ArbiterOutput:
        """
        把 calibration 结果回填到最终输出字段

        这是另一个关键闭环：calibration 不仅仅是诊断，而是真正影响最终输出
        """
        # Update acceptance likelihood
        arbiter_output.acceptance_likelihood = calibration.acceptance_likelihood

        # Update calibrated rating
        if calibration.calibrated_rating is not None:
            arbiter_output.calibrated_rating = calibration.calibrated_rating

        # Add borderline likelihood if available (three-way calibration)
        if calibration.borderline_likelihood is not None:
            arbiter_output.trace["borderline_likelihood"] = calibration.borderline_likelihood
            arbiter_output.trace["rejection_likelihood"] = calibration.rejection_likelihood

        # Derive final decision from calibration if confidence is high
        if calibration.calibration_confidence and calibration.calibration_confidence > 0.7:
            if calibration.acceptance_likelihood and calibration.acceptance_likelihood > 0.6:
                # Don't override, but note the calibration suggests accept
                arbiter_output.trace["calibration_suggestion"] = "accept"
            elif calibration.rejection_likelihood and calibration.rejection_likelihood > 0.6:
                arbiter_output.trace["calibration_suggestion"] = "reject"

        return arbiter_output
