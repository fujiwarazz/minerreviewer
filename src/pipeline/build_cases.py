"""从 papers + reviews 构建 PaperCase"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from clients.llm_client import LLMClient, LLMConfig
from common.types import Paper, PaperCase, PaperSignature, Review
from common.utils import read_yaml
from pipeline.parse_paper import PaperParser
from storage.case_store import CaseStore
from storage.doc_store import DocStore

logger = logging.getLogger(__name__)


class CaseBuilder:
    """从 papers + reviews 构建 PaperCase"""

    def __init__(
        self,
        llm: LLMClient,
        embedding_client: EmbeddingClient,
        doc_store: DocStore,
        case_store: CaseStore,
    ) -> None:
        self.llm = llm
        self.embedding_client = embedding_client
        self.doc_store = doc_store
        self.case_store = case_store
        self.parser = PaperParser(llm)

    def build_case(
        self,
        paper: Paper,
        reviews: list[Review],
        parse_signature: bool = True,
    ) -> PaperCase:
        """从论文和审稿构建单个 PaperCase"""
        # Parse paper signature
        signature: PaperSignature | None = None
        if parse_signature:
            try:
                signature = self.parser.parse(paper)
            except Exception as e:
                logger.warning("Failed to parse signature for paper %s: %s", paper.paper_id, e)

        # Aggregate reviews
        top_strengths = self._extract_strengths(reviews)
        top_weaknesses = self._extract_weaknesses(reviews)
        decisive_issues = self._extract_decisive_issues(reviews)
        review_consensus = self._get_consensus(reviews)
        decision = self._get_majority_decision(reviews)
        rating = self._get_mean_rating(reviews)
        transferable_criteria = self._extract_transferable_criteria(reviews)
        failure_patterns = self._extract_failure_patterns(reviews)

        case = PaperCase(
            case_id=str(uuid.uuid4()),
            paper_id=paper.paper_id,
            venue_id=paper.venue_id,
            year=paper.year,
            title=paper.title,
            abstract=paper.abstract,
            paper_signature=signature,
            top_strengths=top_strengths,
            top_weaknesses=top_weaknesses,
            decisive_issues=decisive_issues,
            review_consensus=review_consensus,
            decision=decision,
            rating=rating,
            source_review_ids=[r.review_id for r in reviews],
            transferable_criteria=transferable_criteria,
            failure_patterns=failure_patterns,
        )
        return case

    def build_cases_for_venue(
        self,
        venue_id: str,
        target_year: int | None = None,
        limit: int | None = None,
        skip_existing: bool = True,
    ) -> list[PaperCase]:
        """为某个 venue 构建所有 PaperCase"""
        papers = self.doc_store.load_papers(venue_id)
        reviews = self.doc_store.load_reviews(venue_id)

        if target_year:
            papers = [p for p in papers if p.year is not None and p.year < target_year]
            reviews = [r for r in reviews if r.year is None or r.year < target_year]

        if limit:
            papers = papers[:limit]

        existing_ids = {c.paper_id for c in self.case_store.list_cases(venue_id=venue_id)}

        cases: list[PaperCase] = []
        for paper in papers:
            if skip_existing and paper.paper_id in existing_ids:
                continue
            paper_reviews = [r for r in reviews if r.paper_id == paper.paper_id]
            if not paper_reviews:
                continue
            try:
                case = self.build_case(paper, paper_reviews)
                self.case_store.add_case(case)
                cases.append(case)
                logger.info("Built case for paper %s", paper.paper_id)
            except Exception as e:
                logger.error("Failed to build case for paper %s: %s", paper.paper_id, e)

        return cases

    def _extract_strengths(self, reviews: list[Review]) -> list[str]:
        """从审稿中提取 strengths"""
        strengths: list[str] = []
        for review in reviews:
            text = review.text.lower()
            if "strength" in text or "strengths" in text:
                # Simple extraction: look for lines after "strengths"
                lines = review.text.split("\n")
                in_strengths = False
                for line in lines:
                    line_lower = line.lower().strip()
                    if "strength" in line_lower and (":" in line_lower or line_lower.startswith("strength")):
                        in_strengths = True
                        continue
                    if in_strengths:
                        if line.strip().startswith("-") or line.strip().startswith("*"):
                            strengths.append(line.strip().lstrip("-* ").strip())
                        elif "weakness" in line_lower:
                            break
        return list(set(strengths))[:5]

    def _extract_weaknesses(self, reviews: list[Review]) -> list[str]:
        """从审稿中提取 weaknesses"""
        weaknesses: list[str] = []
        for review in reviews:
            text = review.text.lower()
            if "weakness" in text or "weaknesses" in text:
                lines = review.text.split("\n")
                in_weaknesses = False
                for line in lines:
                    line_lower = line.lower().strip()
                    if "weakness" in line_lower and (":" in line_lower or line_lower.startswith("weakness")):
                        in_weaknesses = True
                        continue
                    if in_weaknesses:
                        if line.strip().startswith("-") or line.strip().startswith("*"):
                            weaknesses.append(line.strip().lstrip("-* ").strip())
                        elif any(kw in line_lower for kw in ["question", "summary", "recommendation"]):
                            break
        return list(set(weaknesses))[:5]

    def _extract_decisive_issues(self, reviews: list[Review]) -> list[str]:
        """从审稿中提取决定性问题"""
        issues: list[str] = []
        for review in reviews:
            # Look for phrases indicating critical issues
            text = review.text
            keywords = ["critical", "major", "fatal", "severe", "fundamental", "crucial"]
            lines = text.split("\n")
            for line in lines:
                line_lower = line.lower()
                if any(kw in line_lower for kw in keywords):
                    issues.append(line.strip())
        return list(set(issues))[:3]

    def _get_consensus(self, reviews: list[Review]) -> str | None:
        """获取审稿共识"""
        if not reviews:
            return None
        decisions = [r.decision for r in reviews if r.decision]
        if not decisions:
            return None
        accepts = sum(1 for d in decisions if d and d.lower().startswith("accept"))
        rejects = sum(1 for d in decisions if d and d.lower().startswith("reject"))
        if accepts > rejects:
            return "accept"
        elif rejects > accepts:
            return "reject"
        else:
            return "borderline"

    def _get_majority_decision(self, reviews: list[Review]) -> str | None:
        """获取多数决策"""
        return self._get_consensus(reviews)

    def _get_mean_rating(self, reviews: list[Review]) -> float | None:
        """获取平均评分"""
        ratings = [r.rating for r in reviews if r.rating is not None]
        if not ratings:
            return None
        return sum(ratings) / len(ratings)

    def _extract_transferable_criteria(self, reviews: list[Review]) -> list[str]:
        """提取可迁移的审稿标准"""
        criteria: list[str] = []
        for review in reviews:
            # Use LLM to extract criteria if available
            lines = review.text.split("\n")
            for line in lines:
                line = line.strip()
                if len(line) > 20 and any(kw in line.lower() for kw in ["should", "could", "would", "needs to", "lacks"]):
                    criteria.append(line)
        return list(set(criteria))[:5]

    def _extract_failure_patterns(self, reviews: list[Review]) -> list[str]:
        """提取失败模式"""
        patterns: list[str] = []
        for review in reviews:
            if review.decision and review.decision.lower().startswith("reject"):
                lines = review.text.split("\n")
                for line in lines:
                    if any(kw in line.lower() for kw in ["missing", "lacks", "insufficient", "incomplete", "unclear"]):
                        patterns.append(line.strip())
        return list(set(patterns))[:3]


def build_cases_command(
    config_path: str,
    venue_id: str | None = None,
    target_year: int | None = None,
    limit: int | None = None,
    skip_existing: bool = True,
) -> None:
    """CLI 命令: 构建 PaperCase"""
    config = read_yaml(config_path)
    venue = venue_id or config.get("venue_id", "ICLR")

    llm = LLMClient(LLMConfig(**config["llm"]))
    embedding_client = EmbeddingClient(EmbeddingConfig(**config["embedding"]))
    doc_store = DocStore()
    case_store = CaseStore(
        config.get("memory", {}).get("case_store_path", "data/processed/cases.jsonl"),
        embedding_client=embedding_client,
    )

    builder = CaseBuilder(llm, embedding_client, doc_store, case_store)
    cases = builder.build_cases_for_venue(
        venue_id=venue,
        target_year=target_year,
        limit=limit,
        skip_existing=skip_existing,
    )
    logger.info("Built %d cases for venue %s", len(cases), venue)