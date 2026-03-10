from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from common.types import CalibrationArtifact, Review
from common.utils import write_json

logger = logging.getLogger(__name__)


class Calibrator:
    def __init__(self, venue_id: str, output_dir: str = "data/processed") -> None:
        self.venue_id = venue_id
        self.output_dir = Path(output_dir)
        self.model_path = self.output_dir / f"calibrator__{venue_id.replace('/', '_')}.joblib"
        self.meta_path = self.output_dir / f"calibrator__{venue_id.replace('/', '_')}.json"
        self.model: IsotonicRegression | None = None

    def fit(self, reviews: list[Review]) -> CalibrationArtifact | None:
        data = [(r.rating, r.decision) for r in reviews if r.rating is not None and r.decision is not None]
        if len(data) < 2:
            logger.warning("Not enough data to fit calibrator")
            return None
        ratings = np.array([x[0] for x in data], dtype=float)
        decisions = np.array([1.0 if str(x[1]).lower().startswith("accept") else 0.0 for x in data])
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(ratings, decisions)
        self.model = model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        artifact = CalibrationArtifact(
            venue_id=self.venue_id,
            method="isotonic",
            trained_at=datetime.utcnow(),
            rating_bins=list(model.X_thresholds_) if hasattr(model, "X_thresholds_") else [],
            acceptance_rates=list(model.y_thresholds_) if hasattr(model, "y_thresholds_") else [],
        )
        write_json(self.meta_path, artifact.model_dump())
        return artifact

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError("Calibrator model not found")
        self.model = joblib.load(self.model_path)

    def predict(self, rating: float) -> float:
        if self.model is None:
            self.load()
        if self.model is None:
            raise RuntimeError("Calibrator not loaded")
        return float(self.model.predict([rating])[0])
