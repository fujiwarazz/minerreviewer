from __future__ import annotations

from pathlib import Path

from common.types import Paper, Review, VenuePolicy
from common.utils import read_json, write_json


class DocStore:
    def __init__(self, root: str | Path = "data/processed") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_papers(self, venue_id: str, papers: list[Paper]) -> Path:
        path = self.root / f"papers__{venue_id.replace('/', '_')}.json"
        write_json(path, [paper.model_dump() for paper in papers])
        return path

    def save_reviews(self, venue_id: str, reviews: list[Review]) -> Path:
        path = self.root / f"reviews__{venue_id.replace('/', '_')}.json"
        write_json(path, [review.model_dump() for review in reviews])
        return path

    def save_policy(self, venue_id: str, policy: VenuePolicy | None) -> Path | None:
        if policy is None:
            return None
        path = self.root / f"policy__{venue_id.replace('/', '_')}.json"
        write_json(path, policy.model_dump())
        return path

    def load_papers(self, venue_id: str) -> list[Paper]:
        path = self.root / f"papers__{venue_id.replace('/', '_')}.json"
        data = read_json(path)
        return [Paper(**item) for item in data]

    def load_reviews(self, venue_id: str) -> list[Review]:
        path = self.root / f"reviews__{venue_id.replace('/', '_')}.json"
        data = read_json(path)
        return [Review(**item) for item in data]

    def load_policy(self, venue_id: str) -> VenuePolicy | None:
        path = self.root / f"policy__{venue_id.replace('/', '_')}.json"
        if not path.exists():
            return None
        data = read_json(path)
        return VenuePolicy(**data)
