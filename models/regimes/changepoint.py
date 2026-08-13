"""Heuristic mean-shift screening for a univariate time series."""

from __future__ import annotations

import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BayesianChangePoint:
    """Compatibility wrapper for a non-Bayesian standardized mean-shift detector."""

    def __init__(self, min_distance: int = 2, percentile: float = 95.0):
        if min_distance < 1:
            raise ValueError("min_distance must be positive.")
        if not 0 < percentile < 100:
            raise ValueError("percentile must be between zero and 100.")
        self.min_distance = min_distance
        self.percentile = percentile
        self.changepoints: list[int] = []
        self.scores_: np.ndarray | None = None

    def detect(self, series: np.ndarray) -> list[int]:
        """Identify locally maximal standardized mean-shift scores above a percentile threshold."""
        scores = self.get_scores(series)
        valid = scores > 0
        if not valid.any():
            self.changepoints = []
            return []
        threshold = float(np.percentile(scores[valid], self.percentile))
        points = []
        for index in range(self.min_distance, len(scores) - self.min_distance):
            local = scores[max(0, index - self.min_distance) : min(len(scores), index + self.min_distance + 1)]
            if scores[index] >= threshold and scores[index] == np.max(local):
                points.append(index)
        self.changepoints = points
        return points

    def get_scores(self, series: np.ndarray) -> np.ndarray:
        """Return non-negative standardized mean-shift scores; these are not posterior probabilities."""
        values = np.asarray(series, dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("Change-point input must be finite.")
        scores = np.zeros(len(values), dtype=float)
        if len(values) < 2 * self.min_distance + 1:
            self.scores_ = scores
            return scores
        for index in range(self.min_distance, len(values) - self.min_distance):
            left, right = values[:index], values[index:]
            pooled_variance = left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right)
            if np.isfinite(pooled_variance) and pooled_variance > 0:
                scores[index] = (left.mean() - right.mean()) ** 2 / pooled_variance
        self.scores_ = scores
        return scores

    def get_probabilities(self, series: np.ndarray) -> np.ndarray:
        """Return normalized change scores retained for backward compatibility."""
        scores = self.get_scores(series)
        total = scores.sum()
        return scores / total if total > 0 else scores
