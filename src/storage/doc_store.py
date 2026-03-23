"""DocStore: 论文和评论的持久化存储

优化版本：
- 添加缓存避免重复加载大文件
- papers__ICLR.json 有 1.4 GB，每次加载需要 8s
- 缓存后只在第一次加载
"""
from __future__ import annotations

import logging
from pathlib import Path

from common.types import Paper, Review, VenuePolicy
from common.utils import read_json, write_json

logger = logging.getLogger(__name__)


class DocStore:
    """论文和评论的持久化存储，支持缓存"""

    def __init__(self, root: str | Path = "data/processed") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # 缓存
        self._papers_cache: dict[str, list[Paper]] = {}
        self._reviews_cache: dict[str, list[Review]] = {}
        self._policy_cache: dict[str, VenuePolicy | None] = {}

    def save_papers(self, venue_id: str, papers: list[Paper]) -> Path:
        path = self.root / f"papers__{venue_id.replace('/', '_')}.json"
        write_json(path, [paper.model_dump() for paper in papers])
        self._papers_cache[venue_id] = papers  # 更新缓存
        return path

    def save_reviews(self, venue_id: str, reviews: list[Review]) -> Path:
        path = self.root / f"reviews__{venue_id.replace('/', '_')}.json"
        write_json(path, [review.model_dump() for review in reviews])
        self._reviews_cache[venue_id] = reviews  # 更新缓存
        return path

    def save_policy(self, venue_id: str, policy: VenuePolicy | None) -> Path | None:
        if policy is None:
            return None
        path = self.root / f"policy__{venue_id.replace('/', '_')}.json"
        write_json(path, policy.model_dump())
        self._policy_cache[venue_id] = policy  # 更新缓存
        return path

    def load_papers(self, venue_id: str) -> list[Paper]:
        # 检查缓存
        if venue_id in self._papers_cache:
            return self._papers_cache[venue_id]

        path = self.root / f"papers__{venue_id.replace('/', '_')}.json"
        if not path.exists():
            return []

        logger.info("Loading papers from %s (%.1f MB)...", path, path.stat().st_size / 1024 / 1024)
        data = read_json(path)
        papers = [Paper(**item) for item in data]
        self._papers_cache[venue_id] = papers  # 缓存
        logger.info("Loaded %d papers (cached)", len(papers))
        return papers

    def load_reviews(self, venue_id: str) -> list[Review]:
        # 检查缓存
        if venue_id in self._reviews_cache:
            return self._reviews_cache[venue_id]

        path = self.root / f"reviews__{venue_id.replace('/', '_')}.json"
        if not path.exists():
            return []

        data = read_json(path)
        reviews = [Review(**item) for item in data]
        self._reviews_cache[venue_id] = reviews  # 缓存
        return reviews

    def load_policy(self, venue_id: str) -> VenuePolicy | None:
        # 检查缓存
        if venue_id in self._policy_cache:
            return self._policy_cache[venue_id]

        path = self.root / f"policy__{venue_id.replace('/', '_')}.json"
        if not path.exists():
            self._policy_cache[venue_id] = None
            return None
        data = read_json(path)
        policy = VenuePolicy(**data)
        self._policy_cache[venue_id] = policy  # 缓存
        return policy

    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._papers_cache.clear()
        self._reviews_cache.clear()
        self._policy_cache.clear()
        logger.info("DocStore cache cleared")