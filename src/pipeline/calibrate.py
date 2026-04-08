from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from common.types import CalibrationArtifact, CalibrationResult, Review
from common.utils import write_json

logger = logging.getLogger(__name__)


class Calibrator:
    """多路校准器，支持 ordinal/three_way/binary 模式"""

    def __init__(
        self,
        venue_id: str,
        output_dir: str = "data/processed",
        mode: str = "ordinal",  # ordinal/three_way/binary
        borderline_low: float = 4.0,
        borderline_high: float = 6.0,
    ) -> None:
        self.venue_id = venue_id
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.borderline_low = borderline_low
        self.borderline_high = borderline_high
        self.model_path = self.output_dir / f"calibrator__{venue_id.replace('/', '_')}.joblib"
        self.meta_path = self.output_dir / f"calibrator__{venue_id.replace('/', '_')}.json"
        self.model: IsotonicRegression | None = None
        # Three-way calibration models
        self.model_accept: IsotonicRegression | None = None
        self.model_borderline: IsotonicRegression | None = None
        self.model_reject: IsotonicRegression | None = None

    def fit(self, reviews: list[Review]) -> CalibrationArtifact | None:
        data = [(r.rating, r.decision) for r in reviews if r.rating is not None and r.decision is not None]
        if len(data) < 2:
            logger.warning("Not enough data to fit calibrator")
            return None
        ratings = np.array([x[0] for x in data], dtype=float)
        decisions = [str(x[1]).lower() for x in data]

        if self.mode in ("three_way", "ordinal"):
            return self._fit_three_way(ratings, decisions)
        else:
            return self._fit_binary(ratings, decisions)

    def _fit_binary(self, ratings: np.ndarray, decisions: list[str]) -> CalibrationArtifact:
        """二值校准 (accept vs non-accept)"""
        binary_labels = np.array([1.0 if d.startswith("accept") else 0.0 for d in decisions])
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(ratings, binary_labels)
        self.model = model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        artifact = CalibrationArtifact(
            venue_id=self.venue_id,
            method=f"isotonic_{self.mode}",
            trained_at=datetime.utcnow(),
            rating_bins=list(model.X_thresholds_) if hasattr(model, "X_thresholds_") else [],
            acceptance_rates=list(model.y_thresholds_) if hasattr(model, "y_thresholds_") else [],
        )
        write_json(self.meta_path, artifact.model_dump())
        return artifact

    def _fit_three_way(self, ratings: np.ndarray, decisions: list[str]) -> CalibrationArtifact:
        """三路校准 (accept/borderline/reject)"""
        # Create three binary classifiers
        accept_labels = np.array([1.0 if d.startswith("accept") else 0.0 for d in decisions])
        reject_labels = np.array([1.0 if d.startswith("reject") else 0.0 for d in decisions])
        borderline_labels = np.array([1.0 if "borderline" in d or "revise" in d else 0.0 for d in decisions])

        # Fit models
        self.model_accept = IsotonicRegression(out_of_bounds="clip")
        self.model_accept.fit(ratings, accept_labels)

        self.model_reject = IsotonicRegression(out_of_bounds="clip")
        self.model_reject.fit(ratings, reject_labels)

        self.model_borderline = IsotonicRegression(out_of_bounds="clip")
        self.model_borderline.fit(ratings, borderline_labels)

        # Also fit a combined model for backwards compatibility
        self.model = self.model_accept

        self.output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "accept": self.model_accept,
            "reject": self.model_reject,
            "borderline": self.model_borderline,
        }, self.model_path)

        artifact = CalibrationArtifact(
            venue_id=self.venue_id,
            method="isotonic_three_way",
            trained_at=datetime.utcnow(),
            rating_bins=[],
            acceptance_rates=[],
        )
        write_json(self.meta_path, artifact.model_dump())
        return artifact

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError("Calibrator model not found")
        loaded = joblib.load(self.model_path)
        if isinstance(loaded, dict):
            # Three-way model
            self.model_accept = loaded.get("accept")
            self.model_reject = loaded.get("reject")
            self.model_borderline = loaded.get("borderline")
            self.model = self.model_accept
        else:
            # Binary model
            self.model = loaded

    def predict(self, rating: float) -> float:
        """预测 acceptance likelihood（向后兼容）"""
        if self.model is None:
            self.load()
        if self.model is None:
            raise RuntimeError("Calibrator not loaded")
        return float(self.model.predict([rating])[0])

    def calibrate(self, rating: float) -> CalibrationResult:
        """
        多路校准，返回 CalibrationResult

        Args:
            rating: 原始评分

        Returns:
            CalibrationResult 包含多路概率
        """
        # Try to load models if not already loaded
        if self.model_accept is None or self.model_reject is None or self.model_borderline is None:
            try:
                self.load()
            except FileNotFoundError:
                pass  # Will fall back to binary or raise error

        # Support both "ordinal" and "three_way" as aliases for three-way calibration
        if self.mode in ("three_way", "ordinal") and self.model_accept and self.model_reject and self.model_borderline:
            return self._calibrate_three_way(rating)
        else:
            return self._calibrate_binary(rating)

    def _calibrate_binary(self, rating: float) -> CalibrationResult:
        """二值校准"""
        if self.model is None:
            try:
                self.load()
            except FileNotFoundError:
                return CalibrationResult(
                    calibrated_rating=rating,
                    acceptance_likelihood=0.5,
                    borderline_likelihood=None,
                    rejection_likelihood=0.5,
                    calibration_confidence=0.0,
                    method="binary",
                )

        acceptance = float(self.model.predict([rating])[0])
        return CalibrationResult(
            calibrated_rating=rating,
            acceptance_likelihood=acceptance,
            borderline_likelihood=None,
            rejection_likelihood=1.0 - acceptance,
            calibration_confidence=0.5,
            method="binary",
        )

    def _calibrate_three_way(self, rating: float) -> CalibrationResult:
        """三路校准"""
        if self.model_accept is None or self.model_reject is None or self.model_borderline is None:
            try:
                self.load()
            except FileNotFoundError:
                return self._calibrate_binary(rating)

        accept_prob = float(self.model_accept.predict([rating])[0])
        reject_prob = float(self.model_reject.predict([rating])[0])
        borderline_prob = float(self.model_borderline.predict([rating])[0])

        # Normalize probabilities
        total = accept_prob + reject_prob + borderline_prob
        if total > 0:
            accept_prob /= total
            reject_prob /= total
            borderline_prob /= total

        # Determine calibrated rating based on probabilities
        calibrated_rating = rating
        if accept_prob > reject_prob and accept_prob > borderline_prob:
            calibrated_rating = rating + (accept_prob * 2 - 1) * 1.0
        elif reject_prob > accept_prob and reject_prob > borderline_prob:
            calibrated_rating = rating - (reject_prob * 2 - 1) * 1.0

        return CalibrationResult(
            calibrated_rating=max(1.0, min(10.0, calibrated_rating)),
            acceptance_likelihood=accept_prob,
            borderline_likelihood=borderline_prob,
            rejection_likelihood=reject_prob,
            calibration_confidence=max(accept_prob, reject_prob, borderline_prob),
            method="three_way",
        )
