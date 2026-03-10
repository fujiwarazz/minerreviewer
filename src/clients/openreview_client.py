from __future__ import annotations

import logging
from typing import Any

import openreview

from common.types import Paper, Review, VenuePolicy

logger = logging.getLogger(__name__)


class OpenReviewClient:
    def __init__(self, baseurl: str = "https://api2.openreview.net") -> None:
        self.client = openreview.api.OpenReviewClient(baseurl=baseurl)

    def fetch_submissions(self, venue_id: str) -> list[Paper]:
        invitation = f"{venue_id}/-/Submission"
        notes = list(self.client.get_all_notes(invitation=invitation))
        papers: list[Paper] = []
        for note in notes:
            content = note.content
            title = self._unwrap_field(content.get("title", ""))
            abstract = self._unwrap_field(content.get("abstract", ""))
            authors = self._unwrap_field(content.get("authors", [])) or []
            papers.append(
                Paper(
                    paper_id=note.id,
                    title=str(title) if title is not None else "",
                    abstract=str(abstract) if abstract is not None else "",
                    venue_id=venue_id,
                    year=self._infer_year(venue_id, content),
                    authors=list(authors) if isinstance(authors, list) else [],
                )
            )
        logger.info("Fetched %s submissions for %s", len(papers), venue_id)
        return papers

    def fetch_reviews(self, venue_id: str) -> list[Review]:
        invitation = f"{venue_id}/-/Official_Review"
        notes = list(self.client.get_all_notes(invitation=invitation))
        reviews: list[Review] = []
        for note in notes:
            content = note.content
            rating = self._parse_rating(self._unwrap_field(content.get("rating")))
            reviews.append(
                Review(
                    review_id=note.id,
                    paper_id=note.forum,
                    venue_id=venue_id,
                    year=self._infer_year(venue_id, content),
                    rating=rating,
                    text=self._review_text(content),
                    decision=self._unwrap_field(content.get("decision")),
                )
            )
        logger.info("Fetched %s reviews for %s", len(reviews), venue_id)
        return reviews

    def fetch_policy(self, venue_id: str) -> VenuePolicy | None:
        invitation_id = f"{venue_id}/-/Official_Review"
        try:
            invitation = self.client.get_invitation(invitation_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to fetch invitation %s: %s", invitation_id, exc)
            return None
        reply_content = invitation.reply.get("content", {}) if invitation.reply else {}
        rating_field = reply_content.get("rating", {})
        rating_scale = rating_field.get("description") or rating_field.get("value")
        return VenuePolicy(
            venue_id=venue_id,
            year=self._infer_year(venue_id, {}),
            rating_scale=rating_scale,
            review_form_fields=reply_content,
        )

    @staticmethod
    def _infer_year(venue_id: str, content: dict[str, Any]) -> int | None:
        for token in venue_id.split("/"):
            if token.isdigit() and len(token) == 4:
                return int(token)
        year = content.get("year")
        if isinstance(year, int):
            return year
        return None

    @staticmethod
    def _parse_rating(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            parts = value.split(":")
            try:
                return float(parts[0])
            except ValueError:
                return None
        return None

    @staticmethod
    def _review_text(content: dict[str, Any]) -> str:
        fields = ["summary", "strengths", "weaknesses", "review"]
        chunks = []
        for field in fields:
            if field not in content:
                continue
            value = OpenReviewClient._unwrap_field(content.get(field))
            if value:
                chunks.append(str(value))
        return "\n".join(chunks).strip()

    @staticmethod
    def _unwrap_field(value: Any) -> Any:
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value
