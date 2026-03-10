from __future__ import annotations

from common.types import Paper, Review
from pipeline.retrieve import filter_by_year


def test_filter_by_year() -> None:
    items = [
        Paper(paper_id="1", title="a", abstract="", venue_id="v", year=2023),
        Paper(paper_id="2", title="b", abstract="", venue_id="v", year=2024),
        Review(review_id="r1", paper_id="1", venue_id="v", year=2022, text=""),
    ]
    filtered = filter_by_year(items, target_year=2024)
    paper_ids = {item.paper_id for item in filtered if isinstance(item, Paper)}
    assert "1" in paper_ids
    assert "2" not in paper_ids
