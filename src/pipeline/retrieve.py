from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from common.types import ExperienceCard, Paper, PaperCase, PaperSignature, Review, RetrievalBundle
from storage.case_store import CaseStore
from storage.doc_store import DocStore
from storage.faiss_index import FaissIndex
from storage.memory_store import MemoryStore
from storage.milvus_store import MilvusConfig, MilvusStore

logger = logging.getLogger(__name__)


def filter_by_year(items: list[Paper | Review], target_year: int | None) -> list[Paper | Review]:
    if target_year is None:
        return items
    return [item for item in items if item.year is None or item.year < target_year]


class Retriever:
    def __init__(
        self,
        venue_id: str,
        embedding_cfg: EmbeddingConfig,
        vector_store: dict | None = None,
        index_root: str = "data/index",
        case_store: CaseStore | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.venue_id = venue_id
        self.embedding_client = EmbeddingClient(embedding_cfg)
        self.vector_store = vector_store or {}
        self.index_root = Path(index_root)
        self.doc_store = DocStore()
        self.case_store = case_store
        self.memory_store = memory_store

    def retrieve(
        self,
        target_paper: Paper,
        top_k_papers: int,
        top_k_reviews: int,
        unrelated_k: int,
        similarity_threshold: float,
        target_year: int | None,
        paper_signature: PaperSignature | None = None,
        use_case_memory: bool = True,
    ) -> RetrievalBundle:
        papers = self.doc_store.load_papers(self.venue_id)
        reviews = self.doc_store.load_reviews(self.venue_id)
        papers_filtered = [p for p in filter_by_year(papers, target_year) if p.paper_id != target_paper.paper_id]
        reviews_filtered = [r for r in filter_by_year(reviews, target_year) if r.paper_id != target_paper.paper_id]

        backend = self.vector_store.get("backend", "faiss")
        review_index_available = False
        query_vec: np.ndarray | None = None
        if backend == "milvus":
            milvus_cfg = MilvusConfig(
                host=self.vector_store.get("host", "localhost"),
                port=int(self.vector_store.get("port", 19530)),
                papers_collection=self.vector_store.get("papers_collection", f"papers_{self.venue_id.replace('/', '_')}"),
                reviews_collection=self.vector_store.get("reviews_collection", f"reviews_{self.venue_id.replace('/', '_')}"),
            )
            milvus = MilvusStore(milvus_cfg)
            query_vec = self.embedding_client.embed([f"{target_paper.title}\n{target_paper.abstract}"])[0].tolist()
            paper_ids = milvus.search_ids(milvus_cfg.papers_collection, query_vec, top_k_papers)
            review_ids = milvus.search_ids(milvus_cfg.reviews_collection, query_vec, top_k_reviews)
            review_index_available = bool(review_ids)
        else:
            paper_index = FaissIndex(
                self.index_root / f"papers__{self.venue_id.replace('/', '_')}.faiss",
                self.index_root / f"papers__{self.venue_id.replace('/', '_')}.meta.json",
            )
            review_index = FaissIndex(
                self.index_root / f"reviews__{self.venue_id.replace('/', '_')}.faiss",
                self.index_root / f"reviews__{self.venue_id.replace('/', '_')}.meta.json",
            )
            paper_index.load()
            review_index_available = review_index.index_path.exists() and review_index.meta_path.exists()
            if review_index_available:
                review_index.load()
            query_vec = self.embedding_client.embed([f"{target_paper.title}\n{target_paper.abstract}"])
            _, paper_ids = paper_index.search(query_vec, top_k_papers)
            review_ids: list[str] = []
            if review_index_available:
                _, review_ids = review_index.search(query_vec, top_k_reviews)

        paper_lookup = {paper.paper_id: paper for paper in papers_filtered}
        review_lookup = {review.review_id: review for review in reviews_filtered}

        related_papers = [paper_lookup[pid] for pid in paper_ids if pid in paper_lookup]
        related_reviews = [review_lookup[rid] for rid in review_ids if rid in review_lookup]

        unrelated_pool = [p for p in papers_filtered if p.paper_id not in paper_ids]
        if unrelated_pool:
            unrelated_papers = random.sample(unrelated_pool, min(unrelated_k, len(unrelated_pool)))
        else:
            unrelated_papers = []

        policy = self.doc_store.load_policy(self.venue_id)

        # === Multi-channel retrieval ===
        similar_paper_cases: list[PaperCase] = []
        case_scores: list[dict] = []  # 记录分数用于 trace
        supporting_papers: list[Paper] = []
        critique_cases: list[ExperienceCard] = []
        policy_cards: list[ExperienceCard] = []
        failure_cards: list[ExperienceCard] = []

        # 1. Retrieve similar paper cases with hybrid retrieval
        if use_case_memory and self.case_store:
            try:
                query_text = f"{target_paper.title}\n{target_paper.abstract}"
                use_hybrid = self.vector_store.get("case_rerank_enabled", True)
                # Retrieve more candidates for balancing
                # Don't restrict to same venue - allow cross-venue retrieval for better rating anchors
                results = self.case_store.retrieve_cases(
                    query_text=query_text,
                    signature=paper_signature,
                    top_k=min(top_k_papers * 4, 20),  # Get more candidates
                    venue_id=None,  # Allow all venues for better rating diversity
                    use_hybrid=use_hybrid,
                    exclude_paper_id=target_paper.paper_id,  # 避免数据泄露
                    before_year=target_year,  # 只用目标年份之前的案例
                )
                # Balance by decision
                similar_paper_cases = self._balance_cases_by_decision(
                    results, target_count=top_k_papers
                )
                case_scores = [
                    {"case_id": case.case_id, "scores": scores}
                    for case, scores in results
                    if case in similar_paper_cases
                ]
                logger.info(
                    "Retrieved %d similar paper cases (balanced from %d candidates, hybrid=%s)",
                    len(similar_paper_cases),
                    len(results),
                    use_hybrid,
                )
            except Exception as e:
                logger.warning("Failed to retrieve paper cases: %s", e)

        # 2. Supporting papers (keep original related_papers)
        supporting_papers = related_papers

        # 3. Retrieve policy cards from memory
        if self.memory_store:
            try:
                # Get all themes from config or use common themes
                themes = ["quality", "novelty", "clarity", "significance", "reproducibility", "soundness"]
                for theme in themes:
                    cards = self.memory_store.list_active(
                        venue_id=self.venue_id,
                        theme=theme,
                    )
                    policy_cards.extend(cards)
                logger.info("Retrieved %d policy cards", len(policy_cards))
            except Exception as e:
                logger.warning("Failed to retrieve policy cards: %s", e)

        # 4. Retrieve critique cases (from memory with kind=critique)
        if self.memory_store:
            try:
                critique_cases = [
                    card for card in self.memory_store.cards
                    if card.kind == "critique" and card.active and
                    (card.venue_id is None or card.venue_id == self.venue_id)
                ][:top_k_reviews]
                logger.info("Retrieved %d critique cases", len(critique_cases))
            except Exception as e:
                logger.warning("Failed to retrieve critique cases: %s", e)

        # 5. Retrieve failure cards (from memory with kind=failure)
        if self.memory_store:
            try:
                failure_cards = [
                    card for card in self.memory_store.cards
                    if card.kind == "failure" and card.active and
                    (card.venue_id is None or card.venue_id == self.venue_id)
                ][:5]
                logger.info("Retrieved %d failure cards", len(failure_cards))
            except Exception as e:
                logger.warning("Failed to retrieve failure cards: %s", e)

        trace = {
            "paper_ids": paper_ids,
            "review_ids": review_ids,
            "filter_year": target_year,
            "case_ids": [c.case_id for c in similar_paper_cases],
            "case_scores": case_scores,  # 新增：记录 embedding/signature/final 分数
            "policy_card_ids": [c.card_id for c in policy_cards],
            "critique_card_ids": [c.card_id for c in critique_cases],
            "failure_card_ids": [c.card_id for c in failure_cards],
        }
        return RetrievalBundle(
            target_paper=target_paper,
            similar_paper_cases=similar_paper_cases,
            supporting_papers=supporting_papers,
            critique_cases=critique_cases,
            policy_cards=policy_cards,
            failure_cards=failure_cards,
            related_papers=related_papers,
            related_reviews=related_reviews,
            unrelated_papers=unrelated_papers,
            venue_policy=policy,
            trace=trace,
        )

    def retrieve_similar_reviews(self, query_text: str, top_k: int, target_year: int | None, exclude_paper_id: str | None = None) -> list[Review]:
        backend = self.vector_store.get("backend", "faiss")
        if backend == "milvus":
            milvus_cfg = MilvusConfig(
                host=self.vector_store.get("host", "localhost"),
                port=int(self.vector_store.get("port", 19530)),
                papers_collection=self.vector_store.get("papers_collection", f"papers_{self.venue_id.replace('/', '_')}"),
                reviews_collection=self.vector_store.get("reviews_collection", f"reviews_{self.venue_id.replace('/', '_')}"),
            )
            milvus = MilvusStore(milvus_cfg)
            query_vec = self.embedding_client.embed([query_text])[0].tolist()
            review_ids = milvus.search_ids(milvus_cfg.reviews_collection, query_vec, top_k)
        else:
            review_index = FaissIndex(
                self.index_root / f"reviews__{self.venue_id.replace('/', '_')}.faiss",
                self.index_root / f"reviews__{self.venue_id.replace('/', '_')}.meta.json",
            )
            if not review_index.index_path.exists() or not review_index.meta_path.exists():
                return []
            review_index.load()
            query_vec = self.embedding_client.embed([query_text])
            _, review_ids = review_index.search(query_vec, top_k)
        reviews = self.doc_store.load_reviews(self.venue_id)
        reviews = [r for r in filter_by_year(reviews, target_year)]
        if exclude_paper_id:
            reviews = [r for r in reviews if r.paper_id != exclude_paper_id]
        review_lookup = {review.review_id: review for review in reviews}
        return [review_lookup[rid] for rid in review_ids if rid in review_lookup]

    def _similarity(self, query_vec: np.ndarray, paper: Paper) -> float:
        paper_vec = self.embedding_client.embed([f"{paper.title}\n{paper.abstract}"])
        norm_query = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        norm_paper = paper_vec / (np.linalg.norm(paper_vec) + 1e-12)
        score = np.dot(norm_query, norm_paper.T).squeeze()
        return float(score)

    def _balance_cases_by_decision(
        self,
        results: list[tuple[PaperCase, dict]],
        target_count: int = 4,
    ) -> list[PaperCase]:
        """Balance retrieved cases by decision (Accept/Reject).

        Ensures roughly equal representation of Accept and Reject cases
        while preserving similarity ranking within each group.
        """
        if not results:
            return []

        # Separate by decision
        accept_cases = []
        reject_cases = []
        other_cases = []

        for case, scores in results:
            if case.decision == "Accept":
                accept_cases.append((case, scores))
            elif case.decision == "Reject":
                reject_cases.append((case, scores))
            else:
                other_cases.append((case, scores))

        # Target: roughly equal Accept/Reject
        half = target_count // 2
        remainder = target_count % 2

        # Take top from each category (already sorted by similarity)
        selected_accept = [c for c, _ in accept_cases[:half + remainder]]
        selected_reject = [c for c, _ in reject_cases[:half]]

        # Interleave to preserve some ranking
        balanced = []
        for i in range(max(len(selected_accept), len(selected_reject))):
            if i < len(selected_accept):
                balanced.append(selected_accept[i])
            if i < len(selected_reject):
                balanced.append(selected_reject[i])

        # Fill remaining slots with other cases if needed
        if len(balanced) < target_count:
            for case, _ in other_cases:
                if len(balanced) >= target_count:
                    break
                balanced.append(case)

        logger.info(
            "Balanced cases: %d Accept, %d Reject from %d accept candidates, %d reject candidates",
            len(selected_accept),
            len(selected_reject),
            len(accept_cases),
            len(reject_cases),
        )

        return balanced[:target_count]
