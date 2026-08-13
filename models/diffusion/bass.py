"""Bass diffusion model for cumulative technology adoption."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BassDiffusion:
    """Estimate and forecast a constrained Bass diffusion curve."""

    def __init__(self, M: float | None = None, p: float | None = None, q: float | None = None):
        self.M = M
        self.p = p
        self.q = q
        self.fitted = False
        self.covariance_: np.ndarray | None = None
        self.fit_diagnostics_: dict[str, Any] = {}

    @staticmethod
    def cumulative_adoption(t: np.ndarray | float, M: float, p: float, q: float) -> np.ndarray:
        """Evaluate cumulative Bass adoption for non-negative time and parameters."""
        time = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(time)) or np.any(time < 0):
            raise ValueError("Bass time values must be finite and non-negative.")
        if not np.all(np.isfinite([M, p, q])) or M <= 0 or p <= 0 or q < 0:
            raise ValueError("Bass parameters require M > 0, p > 0 and q >= 0.")
        decay = np.exp(-np.clip((p + q) * time, 0, 700))
        return M * (1 - decay) / (1 + (q / p) * decay)

    def fit(self, t: np.ndarray, S: np.ndarray) -> tuple[float, float, float]:
        """Fit M, p and q to a monotone cumulative adoption series."""
        time, adoption = self._validate_training_data(t, S)
        observed_max = float(adoption.max())
        max_time = max(float(time.max()), 1.0)
        m_lower = max(observed_max * (1 + 1e-6), 1e-6)
        m_upper = max(m_lower * 100, observed_max + 1.0)
        p_guess = min(max(0.03 / max_time, 1e-4), 0.25)
        q_guess = min(max(0.5 / max_time, 1e-4), 0.95)
        m_guess = min(max(observed_max * 1.5, m_lower * 1.01), m_upper * 0.9)

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                parameters, covariance = curve_fit(
                    self.cumulative_adoption,
                    time,
                    adoption,
                    p0=[m_guess, p_guess, q_guess],
                    bounds=([m_lower, 1e-8, 0.0], [m_upper, 1.0, 5.0]),
                    maxfev=50_000,
                )
        except (RuntimeError, ValueError, OptimizeWarning) as error:
            raise RuntimeError(f"Bass optimization failed: {error}") from error

        self.M, self.p, self.q = (float(value) for value in parameters)
        self.covariance_ = covariance
        fitted = self.cumulative_adoption(time, self.M, self.p, self.q)
        residuals = adoption - fitted
        self.fit_diagnostics_ = {
            "n_obs": int(len(adoption)),
            "sse": float(np.square(residuals).sum()),
            "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
            "observed_max": observed_max,
            "market_potential_bound": m_lower,
        }
        self.fitted = True
        return self.M, self.p, self.q

    def predict(self, t: np.ndarray) -> np.ndarray:
        """Forecast cumulative adoption at the supplied time values."""
        if not self.fitted or None in (self.M, self.p, self.q):
            raise ValueError("Model is not fitted. Call fit() first.")
        return self.cumulative_adoption(t, self.M, self.p, self.q)

    def get_params(self) -> dict[str, float]:
        """Return the fitted Bass parameters."""
        if not self.fitted or None in (self.M, self.p, self.q):
            raise ValueError("Model is not fitted. Call fit() first.")
        return {"M": float(self.M), "p": float(self.p), "q": float(self.q)}

    @staticmethod
    def _validate_training_data(t: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        time = np.asarray(t, dtype=float).reshape(-1)
        adoption = np.asarray(S, dtype=float).reshape(-1)
        if len(time) != len(adoption) or len(time) < 4:
            raise ValueError("Bass fitting requires at least four aligned observations.")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(adoption)):
            raise ValueError("Bass inputs must be finite.")
        if np.any(time < 0) or np.any(np.diff(time) <= 0):
            raise ValueError("Bass time values must be strictly increasing and non-negative.")
        if np.any(adoption < 0) or np.any(np.diff(adoption) < 0):
            raise ValueError("Cumulative adoption must be non-negative and non-decreasing.")
        if adoption[-1] <= 0:
            raise ValueError("Cumulative adoption must contain a positive observation.")
        return time, adoption
