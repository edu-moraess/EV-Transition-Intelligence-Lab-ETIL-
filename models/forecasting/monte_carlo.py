"""Monte Carlo helpers for probabilistic forecast scenarios."""

from __future__ import annotations

import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


class MonteCarloSimulator:
    """Generate reproducible empirical forecast distributions."""

    def __init__(
        self,
        n_simulations: int = 5_000,
        horizon: int = 10,
        confidence_level: float = 0.95,
        seed: int | None = 42,
    ):
        if n_simulations < 1 or horizon < 1:
            raise ValueError("n_simulations and horizon must be positive.")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one.")
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.confidence_level = confidence_level
        self.seed = seed
        self.simulations: np.ndarray | None = None

    def simulate_from_gp(self, gp_model, X_future: np.ndarray) -> np.ndarray:
        """Draw correlated posterior trajectories from a fitted GP model."""
        if hasattr(gp_model, "sample_paths"):
            simulations = gp_model.sample_paths(X_future, self.n_simulations, seed=self.seed)
        else:
            mean, std = gp_model.predict(X_future, return_std=True)
            rng = np.random.default_rng(self.seed)
            simulations = rng.normal(mean, std, size=(self.n_simulations, len(X_future)))
        self.simulations = np.asarray(simulations, dtype=float)
        if not np.all(np.isfinite(self.simulations)):
            raise ValueError("Monte Carlo simulation produced non-finite values.")
        return self.simulations

    def get_percentiles(self, simulations: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return empirical lower, median and upper quantiles of simulated paths."""
        values = self.simulations if simulations is None else np.asarray(simulations, dtype=float)
        if values is None or values.ndim != 2 or len(values) == 0:
            raise ValueError("A non-empty two-dimensional simulation array is required.")
        lower_q = (1 - self.confidence_level) * 50
        upper_q = (1 + self.confidence_level) * 50
        return (
            np.percentile(values, lower_q, axis=0),
            np.percentile(values, 50, axis=0),
            np.percentile(values, upper_q, axis=0),
        )

    @staticmethod
    def scenario_adjust(simulations: np.ndarray, scenario: str = "base") -> np.ndarray:
        """Apply an explicitly heuristic scenario multiplier to simulated outcomes."""
        factors = {"base": 1.0, "optimistic": 1.2, "pessimistic": 0.8}
        if scenario not in factors:
            raise ValueError(f"Unknown scenario: {scenario}")
        return np.asarray(simulations, dtype=float) * factors[scenario]
