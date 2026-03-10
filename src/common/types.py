from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    venue_id: str
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    fulltext: str | None = None


class Review(BaseModel):
    review_id: str
    paper_id: str
    venue_id: str
    year: int | None = None
    rating: float | None = None
    text: str
    decision: str | None = None


class VenuePolicy(BaseModel):
    venue_id: str
    year: int | None = None
    rating_scale: str | None = None
    review_form_fields: dict[str, Any] = Field(default_factory=dict)


class Criterion(BaseModel):
    criterion_id: str
    text: str
    theme: str
    kind: Literal["content", "policy"]
    source_ids: list[str] = Field(default_factory=list)


class RetrievalBundle(BaseModel):
    target_paper: Paper
    related_papers: list[Paper]
    related_reviews: list[Review]
    unrelated_papers: list[Paper]
    venue_policy: VenuePolicy | None
    trace: dict[str, Any] = Field(default_factory=dict)


class ThemeOutput(BaseModel):
    theme: str
    strengths: list[str]
    weaknesses: list[str]
    severity_tags: list[str]
    notes: str | None = None
    criteria_used: list[str] = Field(default_factory=list)


class ArbiterOutput(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    raw_rating: float
    decision_recommendation: str | None = None
    raw_decision: str | None = None
    calibrated_rating: float | None = None
    acceptance_likelihood: float | None = None
    trace: dict[str, Any] = Field(default_factory=dict)


class CalibrationArtifact(BaseModel):
    venue_id: str
    method: str
    trained_at: datetime
    rating_bins: list[float] = Field(default_factory=list)
    acceptance_rates: list[float] = Field(default_factory=list)


class ExperienceCard(BaseModel):
    card_id: str
    venue_id: str
    theme: str
    content: str
    kind: Literal["policy"] = "policy"
    version: int = 1
    active: bool = True
    quality: float = 0.5
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_trace: dict[str, Any] = Field(default_factory=dict)
