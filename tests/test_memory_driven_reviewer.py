"""Tests for memory-driven reviewer system components"""
from __future__ import annotations

import pytest

from common.types import (
    ActivatedCriterion,
    ArbiterOutput,
    CalibrationResult,
    DecisionVerificationReport,
    ExperienceCard,
    Paper,
    PaperCase,
    PaperSignature,
    RetrievalBundle,
    ScoreConsistencyReport,
)


class TestPaperSignature:
    """Tests for PaperSignature"""

    def test_create_empty(self):
        sig = PaperSignature()
        assert sig.paper_type is None
        assert sig.tasks == []
        assert sig.domain is None

    def test_create_with_values(self):
        sig = PaperSignature(
            paper_type="empirical",
            tasks=["classification", "detection"],
            domain="vision",
            method_family=["transformer", "cnn"],
        )
        assert sig.paper_type == "empirical"
        assert len(sig.tasks) == 2
        assert sig.domain == "vision"


class TestPaperCase:
    """Tests for PaperCase"""

    def test_create_paper_case(self):
        case = PaperCase(
            case_id="test-case-1",
            paper_id="paper-123",
            venue_id="ICLR",
            year=2024,
            title="Test Paper",
            abstract="Test abstract",
            decision="accept",
            rating=7.5,
        )
        assert case.case_id == "test-case-1"
        assert case.decision == "accept"
        assert case.rating == 7.5
        assert case.top_strengths == []


class TestExperienceCard:
    """Tests for ExperienceCard"""

    def test_create_policy_card(self):
        card = ExperienceCard(
            card_id="card-1",
            kind="policy",
            scope="venue",
            venue_id="ICLR",
            theme="quality",
            content="Test policy content",
            utility=0.7,
            confidence=0.8,
        )
        assert card.kind == "policy"
        assert card.scope == "venue"
        assert card.utility == 0.7

    def test_create_failure_card(self):
        card = ExperienceCard(
            card_id="card-2",
            kind="failure",
            scope="venue",
            venue_id="ICLR",
            theme="reproducibility",
            content="Missing implementation details",
            trigger=["code unavailable"],
        )
        assert card.kind == "failure"
        assert card.trigger == ["code unavailable"]


class TestActivatedCriterion:
    """Tests for ActivatedCriterion"""

    def test_create_activated_criterion(self):
        ac = ActivatedCriterion(
            theme="novelty",
            criterion="The method lacks novelty compared to existing approaches",
            source="memory",
            priority=5,
            trigger_reason="Retrieved from memory",
        )
        assert ac.theme == "novelty"
        assert ac.source == "memory"
        assert ac.priority == 5


class TestScoreConsistencyReport:
    """Tests for ScoreConsistencyReport"""

    def test_create_consistency_report(self):
        report = ScoreConsistencyReport(
            similar_review_count=5,
            mean_rating=6.5,
            median_rating=7.0,
            rating_deviation=1.2,
            decision_distribution={"accept": 3, "reject": 2},
            consistency_level="high",
            justification_needed=False,
        )
        assert report.similar_review_count == 5
        assert report.consistency_level == "high"
        assert report.warning is None


class TestDecisionVerificationReport:
    """Tests for DecisionVerificationReport"""

    def test_create_verification_report(self):
        report = DecisionVerificationReport(
            passed=True,
            score_text_alignment="aligned",
            evidence_support_level="strong",
            venue_alignment_level="high",
            warnings=[],
            requires_revision=False,
        )
        assert report.passed is True
        assert report.score_text_alignment == "aligned"

    def test_failed_verification(self):
        report = DecisionVerificationReport(
            passed=False,
            score_text_alignment="misaligned",
            evidence_support_level="weak",
            venue_alignment_level="low",
            warnings=["Score does not match text description"],
            requires_revision=True,
        )
        assert report.passed is False
        assert len(report.warnings) == 1


class TestCalibrationResult:
    """Tests for CalibrationResult"""

    def test_binary_calibration(self):
        result = CalibrationResult(
            calibrated_rating=7.5,
            acceptance_likelihood=0.75,
            rejection_likelihood=0.25,
            method="binary",
        )
        assert result.acceptance_likelihood == 0.75
        assert result.borderline_likelihood is None

    def test_three_way_calibration(self):
        result = CalibrationResult(
            calibrated_rating=6.0,
            acceptance_likelihood=0.3,
            borderline_likelihood=0.5,
            rejection_likelihood=0.2,
            method="three_way",
        )
        assert result.borderline_likelihood == 0.5


class TestRetrievalBundle:
    """Tests for RetrievalBundle"""

    def test_create_retrieval_bundle(self):
        paper = Paper(
            paper_id="test-1",
            title="Test Paper",
            abstract="Test abstract",
            venue_id="ICLR",
        )
        bundle = RetrievalBundle(
            target_paper=paper,
            similar_paper_cases=[],
            supporting_papers=[],
            critique_cases=[],
            policy_cards=[],
            failure_cards=[],
        )
        assert bundle.target_paper.paper_id == "test-1"
        assert bundle.similar_paper_cases == []
        assert bundle.policy_cards == []

    def test_retrieval_bundle_with_cases(self):
        paper = Paper(
            paper_id="test-1",
            title="Test Paper",
            abstract="Test abstract",
            venue_id="ICLR",
        )
        case = PaperCase(
            case_id="case-1",
            title="Similar Paper",
            abstract="Similar abstract",
        )
        card = ExperienceCard(
            card_id="card-1",
            kind="policy",
            theme="quality",
            content="Test",
        )
        bundle = RetrievalBundle(
            target_paper=paper,
            similar_paper_cases=[case],
            policy_cards=[card],
        )
        assert len(bundle.similar_paper_cases) == 1
        assert len(bundle.policy_cards) == 1


# Tests for new pipeline modules

class TestPaperParser:
    """Tests for PaperParser"""

    def test_parse_paper(self):
        from pipeline.parse_paper import PaperParser

        # Mock LLM client
        class MockLLM:
            def generate_json(self, prompt):
                return {
                    "paper_type": "empirical",
                    "tasks": ["classification"],
                    "domain": "vision",
                }

        parser = PaperParser(MockLLM())
        paper = Paper(
            paper_id="test",
            title="Vision Transformer for Classification",
            abstract="We propose a vision transformer for image classification.",
            venue_id="ICLR",
        )
        signature = parser.parse(paper)
        assert signature.paper_type == "empirical"
        assert "classification" in signature.tasks


class TestCriteriaPlanner:
    """Tests for CriteriaPlanner"""

    def test_plan_criteria(self):
        from pipeline.plan_criteria import CriteriaPlanner

        class MockLLM:
            pass

        planner = CriteriaPlanner(MockLLM())
        paper = Paper(
            paper_id="test",
            title="Test",
            abstract="Test",
            venue_id="ICLR",
        )
        bundle = RetrievalBundle(target_paper=paper)
        activated = planner.plan(
            signature=None,
            bundle=bundle,
            mined_criteria=[],
        )
        assert isinstance(activated, list)


class TestScoreConsistencyChecker:
    """Tests for ScoreConsistencyChecker"""

    def test_check_consistency(self):
        from pipeline.check_score_consistency import ScoreConsistencyChecker

        checker = ScoreConsistencyChecker()
        arbiter_output = ArbiterOutput(
            strengths=["Good"],
            weaknesses=["Bad"],
            raw_rating=7.0,
            decision_recommendation="accept",
        )
        paper = Paper(
            paper_id="test",
            title="Test",
            abstract="Test",
            venue_id="ICLR",
        )
        bundle = RetrievalBundle(target_paper=paper)
        report = checker.check(arbiter_output, bundle)
        assert report.consistency_level == "unknown"  # No similar reviews


class TestDecisionVerifier:
    """Tests for DecisionVerifier"""

    def test_verify_decision(self):
        from pipeline.verify_decision import DecisionVerifier

        class MockLLM:
            pass

        verifier = DecisionVerifier(MockLLM())
        arbiter_output = ArbiterOutput(
            strengths=["Strong method"],
            weaknesses=["Limited experiments"],
            raw_rating=6.0,
            decision_recommendation="borderline",
        )
        paper = Paper(
            paper_id="test",
            title="Test",
            abstract="Test",
            venue_id="ICLR",
        )
        bundle = RetrievalBundle(target_paper=paper)
        report = verifier.verify(arbiter_output, paper, bundle)
        assert report.score_text_alignment in ["aligned", "unclear", "misaligned"]


class TestCalibrator:
    """Tests for Calibrator"""

    def test_calibrate_binary(self):
        from pipeline.calibrate import Calibrator

        calibrator = Calibrator("test-venue", mode="binary")
        result = calibrator.calibrate(7.0)
        # Without trained model, should return defaults
        assert result.calibrated_rating == 7.0
        assert result.method in ["binary", "none"]

    def test_calibrate_three_way(self):
        from pipeline.calibrate import Calibrator

        calibrator = Calibrator("test-venue", mode="three_way")
        result = calibrator.calibrate(5.0)
        # Without trained model, should fallback
        assert result.calibrated_rating == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])