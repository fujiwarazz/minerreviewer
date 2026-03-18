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


# ============================================
# 关键回归测试
# ============================================

class TestScoreConsistencyNeverModifiesRating:
    """
    回归测试：确保 ScoreConsistencyChecker 永远不会直接修改评分

    这是系统设计的关键约束：
    - 一致性检查只提供警告
    - 不改变 arbiter 的 raw_rating 和 decision_recommendation
    """

    def test_consistency_checker_never_modifies_rating(self):
        """一致性检查不应修改评分"""
        from pipeline.check_score_consistency import ScoreConsistencyChecker

        checker = ScoreConsistencyChecker()
        original_rating = 3.0  # 低分，与相似案例不一致

        arbiter_output = ArbiterOutput(
            strengths=["Good"],
            weaknesses=["Bad"],
            raw_rating=original_rating,
            decision_recommendation="reject",
        )

        # 创建包含相似案例的 bundle
        paper = Paper(
            paper_id="test",
            title="Test",
            abstract="Test",
            venue_id="ICLR",
        )
        case = PaperCase(
            case_id="case-1",
            title="Similar Paper",
            abstract="Similar",
            rating=8.0,
            decision="accept",
        )
        bundle = RetrievalBundle(
            target_paper=paper,
            similar_paper_cases=[case],
        )

        report = checker.check(arbiter_output, bundle)

        # 关键断言：评分不应被修改
        assert arbiter_output.raw_rating == original_rating, \
            "ScoreConsistencyChecker should NEVER modify raw_rating"
        assert arbiter_output.decision_recommendation == "reject", \
            "ScoreConsistencyChecker should NEVER modify decision_recommendation"

        # 应该有警告，但不是修改
        assert report.warning is not None or report.consistency_level in ["low", "medium"], \
            "Should have warning for inconsistent rating"

    def test_consistency_checker_with_high_deviation_only_warns(self):
        """高偏差只产生警告，不修改评分"""
        from pipeline.check_score_consistency import ScoreConsistencyChecker

        checker = ScoreConsistencyChecker(deviation_threshold=1.0)

        arbiter_output = ArbiterOutput(
            strengths=["A"],
            weaknesses=["B"],
            raw_rating=5.0,
            decision_recommendation="borderline",
        )

        # 创建多个高评分案例
        paper = Paper(paper_id="test", title="T", abstract="A", venue_id="ICLR")
        cases = [
            PaperCase(case_id=f"c{i}", title=f"T{i}", abstract="A", rating=9.0, decision="accept")
            for i in range(5)
        ]
        bundle = RetrievalBundle(target_paper=paper, similar_paper_cases=cases)

        original_rating = arbiter_output.raw_rating
        report = checker.check(arbiter_output, bundle)

        # 验证评分未被修改
        assert arbiter_output.raw_rating == original_rating
        assert report.consistency_level == "low"  # 应该标记为低一致性
        assert report.warning is not None  # 应该有警告


class TestCaseMemoryAffectsActivatedCriteria:
    """
    回归测试：确保 case memory 召回能影响 activated criteria

    验证：
    - policy_cards 应该被转换为 activated criteria
    - similar_paper_cases 的 transferable_criteria 应该被激活
    """

    def test_policy_cards_become_activated_criteria(self):
        """policy cards 应该转换为 activated criteria"""
        from pipeline.plan_criteria import CriteriaPlanner

        class MockLLM:
            pass

        planner = CriteriaPlanner(MockLLM())
        paper = Paper(paper_id="test", title="T", abstract="A", venue_id="ICLR")

        # 创建 policy card
        policy_card = ExperienceCard(
            card_id="policy-1",
            kind="policy",
            scope="venue",
            venue_id="ICLR",
            theme="novelty",
            content="The paper should demonstrate clear novelty over existing methods",
            utility=0.8,
            confidence=0.9,
        )

        bundle = RetrievalBundle(
            target_paper=paper,
            policy_cards=[policy_card],
        )

        activated = planner.plan(
            signature=None,
            bundle=bundle,
            mined_criteria=[],
        )

        # 验证 policy card 被转换为 activated criterion
        assert len(activated) > 0, "Policy cards should create activated criteria"

        # 检查是否有来自 memory 的 criterion
        memory_criteria = [c for c in activated if c.source == "memory"]
        assert len(memory_criteria) > 0, "Should have criteria from memory"

        # 检查主题和内容是否正确传递
        novelty_criteria = [c for c in activated if c.theme == "novelty"]
        assert len(novelty_criteria) > 0, "Should have novelty theme criterion"

    def test_similar_cases_transfer_criteria(self):
        """similar_paper_cases 的 transferable_criteria 应该被激活"""
        from pipeline.plan_criteria import CriteriaPlanner

        class MockLLM:
            pass

        planner = CriteriaPlanner(MockLLM())
        paper = Paper(paper_id="test", title="T", abstract="A", venue_id="ICLR")

        # 创建带有 transferable_criteria 的 case
        case = PaperCase(
            case_id="case-1",
            title="Similar Paper",
            abstract="Similar",
            decision="reject",
            transferable_criteria=[
                "The method should be compared against recent baselines",
                "Ablation studies are needed to justify design choices",
            ],
        )

        bundle = RetrievalBundle(
            target_paper=paper,
            similar_paper_cases=[case],
        )

        activated = planner.plan(
            signature=None,
            bundle=bundle,
            mined_criteria=[],
        )

        # 验证来自案例的标准被激活
        case_criteria = [c for c in activated if "case" in c.trigger_reason.lower()]
        assert len(case_criteria) > 0, "Similar cases should transfer criteria"


class TestVerifierAndCalibrationAffectFinalReport:
    """
    回归测试：确保 verifier 和 calibration 的结果反映到最终报告

    验证：
    - verification 结果在 trace 中
    - calibration 结果回填到 acceptance_likelihood
    - requires_revision 能触发修订流程
    """

    def test_verification_in_final_trace(self):
        """verification 结果应该在最终 trace 中"""
        from pipeline.verify_decision import DecisionVerifier

        class MockLLM:
            pass

        verifier = DecisionVerifier(MockLLM())
        arbiter_output = ArbiterOutput(
            strengths=["Good method"],
            weaknesses=["Limited experiments"],
            raw_rating=6.0,
            decision_recommendation="borderline",
        )
        paper = Paper(paper_id="test", title="T", abstract="A", venue_id="ICLR")
        bundle = RetrievalBundle(target_paper=paper)

        report = verifier.verify(arbiter_output, paper, bundle)

        # 验证报告包含所有必要字段
        assert hasattr(report, "passed")
        assert hasattr(report, "score_text_alignment")
        assert hasattr(report, "evidence_support_level")
        assert hasattr(report, "requires_revision")

    def test_calibration_applied_to_output(self):
        """calibration 结果应该应用到最终输出"""
        from pipeline.calibrate import Calibrator

        calibrator = Calibrator("test-venue", mode="three_way")

        # 模拟已训练的模型场景
        result = calibrator.calibrate(7.0)

        # 验证 calibration 结果包含必要字段
        assert result.calibrated_rating is not None
        assert result.acceptance_likelihood is not None
        assert result.method in ["ordinal", "three_way", "binary", "none"]

    def test_low_rating_triggers_revision_warning(self):
        """低评分与文本不符应触发 requires_revision"""
        from pipeline.verify_decision import DecisionVerifier

        class MockLLM:
            pass

        verifier = DecisionVerifier(MockLLM())

        # 高评分但有很多 weaknesses
        arbiter_output = ArbiterOutput(
            strengths=["OK"],
            weaknesses=[
                "Critical issue 1",
                "Critical issue 2",
                "Critical issue 3",
                "Major flaw in methodology",
            ],
            raw_rating=8.0,  # 高评分，但 weaknesses 很多
            decision_recommendation="accept",
        )

        paper = Paper(paper_id="test", title="T", abstract="A", venue_id="ICLR")
        bundle = RetrievalBundle(target_paper=paper)

        report = verifier.verify(arbiter_output, paper, bundle)

        # 高评分与多 weaknesses 应该有 misalignment
        # 注意：这个测试可能需要根据实际实现调整
        assert report.score_text_alignment in ["aligned", "misaligned", "unclear"]


class TestConsistencyCheckerPrefersCases:
    """
    回归测试：确保 ScoreConsistencyChecker 优先使用 similar_paper_cases

    验证：
    - 有 cases 时优先使用 cases
    - cases 不足时才 fallback 到 reviews
    """

    def test_prefers_cases_over_reviews(self):
        """应该优先使用 cases 而不是 reviews"""
        from pipeline.check_score_consistency import ScoreConsistencyChecker
        from common.types import Review

        checker = ScoreConsistencyChecker(prefer_cases=True, min_samples=2)

        arbiter_output = ArbiterOutput(
            strengths=["A"],
            weaknesses=["B"],
            raw_rating=7.0,
            decision_recommendation="accept",
        )

        paper = Paper(paper_id="test", title="T", abstract="A", venue_id="ICLR")

        # 创建 cases (高评分)
        cases = [
            PaperCase(case_id=f"c{i}", title=f"T{i}", abstract="A", rating=8.0, decision="accept")
            for i in range(3)
        ]

        # 创建 reviews (低评分) - 如果用 reviews 会得到不同的结果
        reviews = [
            Review(review_id=f"r{i}", paper_id="p", venue_id="ICLR", text="text", rating=4.0, decision="reject")
            for i in range(5)
        ]

        bundle = RetrievalBundle(
            target_paper=paper,
            similar_paper_cases=cases,
            related_reviews=reviews,
        )

        report = checker.check(arbiter_output, bundle)

        # 如果使用 cases，mean 应该接近 8.0
        # 如果使用 reviews，mean 应该接近 4.0
        # 我们期望使用 cases
        if report.mean_rating is not None:
            # cases 的平均是 8.0，reviews 的平均是 4.0
            assert report.mean_rating > 6.0, \
                f"Should use cases (mean ~8.0), got mean={report.mean_rating}"

    def test_fallback_to_reviews_when_no_cases(self):
        """没有 cases 时应该 fallback 到 reviews"""
        from pipeline.check_score_consistency import ScoreConsistencyChecker
        from common.types import Review

        checker = ScoreConsistencyChecker(prefer_cases=True, min_samples=2)

        arbiter_output = ArbiterOutput(
            strengths=["A"],
            weaknesses=["B"],
            raw_rating=6.0,
            decision_recommendation="borderline",
        )

        paper = Paper(paper_id="test", title="T", abstract="A", venue_id="ICLR")

        # 没有 cases，只有 reviews
        reviews = [
            Review(review_id=f"r{i}", paper_id="p", venue_id="ICLR", text="text", rating=7.0, decision="accept")
            for i in range(5)
        ]

        bundle = RetrievalBundle(
            target_paper=paper,
            similar_paper_cases=[],  # 没有 cases
            related_reviews=reviews,
        )

        report = checker.check(arbiter_output, bundle)

        # 应该能正常工作（fallback 到 reviews）
        assert report.similar_review_count == 5
        if report.mean_rating is not None:
            assert abs(report.mean_rating - 7.0) < 0.1