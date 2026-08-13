"""Gaussian-mixture regime classification for EV adoption features."""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from utils.logger import setup_logger

logger = setup_logger(__name__)


class HMMRegimeDetector:
    """Classify observations with a GMM and map components by the first feature."""

    REGIME_NAMES = {0: "Emerging", 1: "Expansion", 2: "Acceleration", 3: "Maturity"}

    def __init__(self, n_states: int = 4, random_seed: int = 42):
        if n_states < 2:
            raise ValueError("n_states must be at least two.")
        self.n_states = n_states
        self.random_seed = random_seed
        self.model: GaussianMixture | None = None
        self.scaler = StandardScaler()
        self.fitted = False
        self._component_order: np.ndarray | None = None
        self._training_states: np.ndarray | None = None
        self.X_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "HMMRegimeDetector":
        """Fit a full-covariance GMM on finite feature vectors."""
        values = self._validate_input(X, minimum_rows=self.n_states)
        self.X_train_ = values.copy()
        scaled = self.scaler.fit_transform(values)
        model = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            n_init=10,
            max_iter=1_000,
            random_state=self.random_seed,
            reg_covar=1e-6,
        )
        model.fit(scaled)
        if not model.converged_:
            raise RuntimeError("Gaussian mixture did not converge.")
        self.model = model
        self._component_order = np.argsort(model.means_[:, 0])
        raw_states = model.predict(scaled)
        self._training_states = self._remap_states(raw_states)
        self.fitted = True
        return self

    def predict_regime(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ordered regime labels and posterior GMM component probabilities."""
        self._check_fitted()
        values = self._validate_input(X, minimum_rows=1)
        scaled = self.scaler.transform(values)
        raw_states = self.model.predict(scaled)
        probabilities = self.model.predict_proba(scaled)[:, self._component_order]
        return self._remap_states(raw_states), probabilities

    def get_regime_name(self, state_id: int) -> str:
        """Return a display label for an ordered GMM component."""
        return self.REGIME_NAMES.get(state_id, f"Regime {state_id}")

    def get_transition_matrix(self) -> np.ndarray:
        """Estimate adjacent-observation transition frequencies from the fitted sequence."""
        self._check_fitted()
        states = self._training_states
        if states is None or len(states) < 2:
            return np.full((self.n_states, self.n_states), np.nan)
        counts = np.zeros((self.n_states, self.n_states), dtype=float)
        for source, target in zip(states[:-1], states[1:]):
            counts[source, target] += 1
        totals = counts.sum(axis=1, keepdims=True)
        return np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals > 0)

    def information_criteria(self) -> dict[str, float]:
        """Return in-sample GMM AIC and BIC for model-selection diagnostics."""
        self._check_fitted()
        if self.X_train_ is None:
            raise ValueError("Training data unavailable.")
        scaled = self.scaler.transform(self.X_train_)
        return {"aic": float(self.model.aic(scaled)), "bic": float(self.model.bic(scaled))}

    def _remap_states(self, raw_states: np.ndarray) -> np.ndarray:
        mapping = {component: rank for rank, component in enumerate(self._component_order)}
        return np.asarray([mapping[int(state)] for state in raw_states], dtype=int)

    def _check_fitted(self) -> None:
        if not self.fitted or self.model is None or self._component_order is None:
            raise ValueError("Model is not fitted. Call fit() first.")

    def _validate_input(self, X: np.ndarray, minimum_rows: int) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[0] < minimum_rows or values.shape[1] < 1:
            raise ValueError(f"Expected a 2D feature array with at least {minimum_rows} rows.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Regime features must be finite.")
        return values
