from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from common.types import Paper, Review, RetrievalBundle
from storage.doc_store import DocStore
from storage.faiss_index import FaissIndex
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
    ) -> None:
        self.venue_id = venue_id
        self.embedding_client = EmbeddingClient(embedding_cfg)
        self.vector_store = vector_store or {}
        self.index_root = Path(index_root)
        self.doc_store = DocStore()

    def retrieve(self, target_paper: Paper, top_k_papers: int, top_k_reviews: int, unrelated_k: int, similarity_threshold: float, target_year: int | None) -> RetrievalBundle:
        papers = self.doc_store.load_papers(self.venue_id)
        reviews = self.doc_store.load_reviews(self.venue_id)
        papers_filtered = [p for p in filter_by_year(papers, target_year) if p.paper_id != target_paper.paper_id]
        reviews_filtered = [r for r in filter_by_year(reviews, target_year) if r.paper_id != target_paper.paper_id]

        backend = self.vector_store.get("backend", "faiss")
        review_index_available = False
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
        trace = {
            "paper_ids": paper_ids,
            "review_ids": review_ids,
            "filter_year": target_year,
        }
        return RetrievalBundle(
            target_paper=target_paper,
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
