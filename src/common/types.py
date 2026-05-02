from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    venue_id: str
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    fulltext: str | None = None


class PaperSignature(BaseModel):
    """论文结构化特征，用于论文分类和相似性匹配"""
    paper_type: str | None = None  # empirical/theoretical/survey/system/analysis
    tasks: list[str] = Field(default_factory=list)  # [classification, generation, detection...]
    domain: str | None = None  # vision/nlp/audio/multimodal/reinforcement_learning
    method_family: list[str] = Field(default_factory=list)  # [transformer, gan, diffusion, gnn...]
    main_claims: list[str] = Field(default_factory=list)
    claim_strength: str | None = None  # strong/moderate/weak
    datasets: list[str] = Field(default_factory=list)
    evaluation_style: list[str] = Field(default_factory=list)  # [ablation, human_eval, benchmark...]
    baseline_coverage: str | None = None  # comprehensive/partial/limited
    risk_profile: list[str] = Field(default_factory=list)  # [reproducibility_risk, ethics_risk...]


class PaperCase(BaseModel):
    """论文记忆案例，存储论文及其审稿关键信息"""
    case_id: str
    paper_id: str | None = None
    venue_id: str | None = None
    year: int | None = None
    title: str
    abstract: str
    paper_signature: PaperSignature | None = None
    primary_area: str | None = None  # 论文研究方向（如 DeepReview-13K 的 primary_area）
    top_strengths: list[str] = Field(default_factory=list)
    top_weaknesses: list[str] = Field(default_factory=list)
    decisive_issues: list[str] = Field(default_factory=list)
    review_consensus: str | None = None
    decision: str | None = None
    rating: float | None = None
    source_review_ids: list[str] = Field(default_factory=list)
    transferable_criteria: list[str] = Field(default_factory=list)
    failure_patterns: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


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
    priority: int = 5  # 新增：用于 Theme Agent 借用时排序


class RetrievalBundle(BaseModel):
    """多通道检索结果"""
    target_paper: Paper
    # 新增：案例和记忆通道
    similar_paper_cases: list[PaperCase] = Field(default_factory=list)
    supporting_papers: list[Paper] = Field(default_factory=list)
    critique_cases: list["ExperienceCard"] = Field(default_factory=list)
    policy_cards: list["ExperienceCard"] = Field(default_factory=list)
    failure_cards: list["ExperienceCard"] = Field(default_factory=list)
    # === 新增：Agent个人记忆 ===
    agent_memories: dict[str, list["ExperienceCard"]] = Field(default_factory=dict)  # 按agent分组的记忆
    agent_memory_scores: dict[str, list[dict]] = Field(default_factory=dict)  # 记忆检索分数
    # 保留原有
    related_papers: list[Paper] = Field(default_factory=list)
    related_reviews: list[Review] = Field(default_factory=list)
    unrelated_papers: list[Paper] = Field(default_factory=list)
    venue_policy: VenuePolicy | None = None
    trace: dict[str, Any] = Field(default_factory=dict)


class ThemeOutput(BaseModel):
    theme: str
    strengths: list[str]
    weaknesses: list[str]
    severity_tags: list[str]
    notes: str | None = None
    criteria_used: list[str] = Field(default_factory=list)


class ArbiterOutput(BaseModel):
    """Arbiter 输出，包含可解释性字段"""
    strengths: list[str]
    weaknesses: list[str]
    raw_rating: float
    decision_recommendation: str | None = None
    raw_decision: str | None = None
    calibrated_rating: float | None = None
    acceptance_likelihood: float | None = None
    # 可解释性字段
    decision_rationale: str | None = None  # 决策理由
    score_rationale: str | None = None  # 评分理由
    key_decisive_issues: list[str] = Field(default_factory=list)  # 决定性问题
    verification_summary: str | None = None  # 验证摘要
    consistency_summary: str | None = None  # 一致性摘要
    trace: dict[str, Any] = Field(default_factory=dict)


class CalibrationArtifact(BaseModel):
    venue_id: str
    method: str
    trained_at: datetime
    rating_bins: list[float] = Field(default_factory=list)
    acceptance_rates: list[float] = Field(default_factory=list)


class ExperienceCard(BaseModel):
    """经验卡片，记录可迁移的评价规律

    kind 类型：
    - strength: 正面评价模式（什么样的论文值得肯定）
    - critique: 批评模式（常见批评角度）
    - failure: 拒稿原因（导致reject的关键问题）
    - policy: 兼容旧类型，自动迁移为strength
    """
    card_id: str
    kind: Literal["strength", "critique", "failure", "policy"] = "strength"
    scope: Literal["global", "venue", "paper_type", "domain"] = "venue"
    venue_id: str | None = None
    theme: str
    content: str
    trigger: list[str] = Field(default_factory=list)  # 触发条件
    utility: float = 0.5
    confidence: float = 0.5
    use_count: int = 0
    source_ids: list[str] = Field(default_factory=list)
    version: int = 1
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_trace: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # === 新增：向量检索和Agent个人记忆支持 ===
    embedding: list[float] | None = None  # 向量表示，用于检索
    owner_agent: str | None = None  # 所属agent（如 "theme_quality", "arbiter"）
    memory_tier: Literal["long_term", "short_term", "ephemeral"] = "long_term"  # 记忆层级
    primary_area: str | None = None  # 所属领域（如 "reinforcement_learning", "generative_models"）

    @field_validator('kind', mode='before')
    @classmethod
    def migrate_policy_kind(cls, v):
        """兼容旧类型：policy自动迁移为strength"""
        if v == "policy":
            return "strength"
        return v

    @staticmethod
    def get_card_text(card: "ExperienceCard") -> str:
        """获取卡片的文本表示，用于embedding"""
        parts = [card.content]
        if card.primary_area:
            parts.insert(0, f"[{card.primary_area}]")
        if card.trigger:
            parts.append("Triggers: " + "; ".join(card.trigger[:3]))
        if card.theme:
            parts.append(f"Theme: {card.theme}")
        return "\n".join(parts)


class ActivatedCriterion(BaseModel):
    """被激活的审稿标准，带优先级和触发条件"""
    theme: str
    criterion: str
    source: str  # memory/mined
    priority: int = 0
    trigger_reason: str = ""
    required_evidence: list[str] = Field(default_factory=list)
    owner_agent: str | None = None


class ScoreConsistencyReport(BaseModel):
    """评分一致性报告"""
    similar_review_count: int = 0
    mean_rating: float | None = None
    median_rating: float | None = None
    rating_deviation: float | None = None
    decision_distribution: dict[str, int] = Field(default_factory=dict)
    consistency_level: str = "unknown"  # high/medium/low/unknown
    warning: str | None = None
    justification_needed: bool = False


class DecisionVerificationReport(BaseModel):
    """决策验证报告"""
    passed: bool
    score_text_alignment: str = "unclear"  # aligned/misaligned/unclear
    evidence_support_level: str = "weak"  # strong/moderate/weak
    venue_alignment_level: str = "unknown"  # high/medium/low/unknown
    warnings: list[str] = Field(default_factory=list)
    requires_revision: bool = False


class CalibrationResult(BaseModel):
    """多路校准结果"""
    calibrated_rating: float | None = None
    acceptance_likelihood: float | None = None
    borderline_likelihood: float | None = None
    rejection_likelihood: float | None = None
    calibration_confidence: float | None = None
    method: str = "ordinal"  # ordinal/three_way/binary
