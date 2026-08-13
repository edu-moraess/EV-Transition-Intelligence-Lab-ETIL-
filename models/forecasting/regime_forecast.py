"""Regime-conditioned Bass forecasts with empirical simulation intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.diffusion.bass import BassDiffusion
from models.regimes.hmm import HMMRegimeDetector
from utils.logger import setup_logger

logger = setup_logger(__name__)

_REGIME_PRIORS = {
    0: {"M_range": (0.7, 2.5), "q_mult": 0.8, "volatility": 0.45, "name": "Emerging"},
    1: {"M_range": (0.8, 1.8), "q_mult": 1.0, "volatility": 0.30, "name": "Expansion"},
    2: {"M_range": (0.9, 1.4), "q_mult": 1.25, "volatility": 0.18, "name": "Acceleration"},
    3: {"M_range": (0.95, 1.1), "q_mult": 0.9, "volatility": 0.10, "name": "Maturity"},
}


class RegimeAwareForecastEngine:
    """Produce heuristic regime-conditioned Bass scenarios for one country."""

    def __init__(self, n_simulations: int = 2_000, seed: int | None = 42):
        if n_simulations < 100:
            raise ValueError("n_simulations must be at least 100.")
        self.n_simulations = n_simulations
        self.seed = seed

    def forecast(self, country_df: pd.DataFrame, horizon: int = 10, regime_id: int | None = None) -> dict:
        """Return regime-specific empirical forecast distributions beyond the last observation."""
        required = {"year", "ev_sales"}
        missing = required - set(country_df.columns)
        if missing:
            return {"error": f"Missing columns: {sorted(missing)}"}
        if len(country_df) < 4 or horizon < 1:
            return {"error": "At least four observations and a positive horizon are required."}
        data = country_df.sort_values("year").reset_index(drop=True)
        if data["ev_sales"].isna().any() or (data["ev_sales"] < 0).any():
            return {"error": "EV sales must be finite and non-negative."}
        if regime_id is None:
            regime_id = self._detect_regime(data)
        if regime_id not in _REGIME_PRIORS:
            return {"error": f"Unknown regime_id: {regime_id}"}

        t_hist = np.arange(1, len(data) + 1, dtype=float)
        cumulative = data["ev_sales"].cumsum().to_numpy(dtype=float)
        try:
            bass = BassDiffusion()
            M, p, q = bass.fit(t_hist, cumulative)
        except (RuntimeError, ValueError) as error:
            return {"error": f"Bass fit failed: {error}"}

        prior = _REGIME_PRIORS[regime_id]
        baseline = self._simulate(M, p, q, t_hist, horizon, prior, int(data["year"].iloc[-1]))
        all_regimes = {
            state: self._simulate(M, p, q, t_hist, horizon, state_prior, int(data["year"].iloc[-1]))
            for state, state_prior in _REGIME_PRIORS.items()
        }
        return {
            "regime_id": regime_id,
            "regime_name": prior["name"],
            "baseline": baseline,
            "all_regimes": all_regimes,
            "bass_params": {"M": M, "p": p, "q": q},
            "prior_applied": prior,
            "n_simulations": self.n_simulations,
            "interval_type": "empirical simulation quantiles",
            "narrative": self._build_narrative(regime_id, prior, M, p, q, baseline),
        }

    def _detect_regime(self, data: pd.DataFrame) -> int:
        features = self._extract_regime_features(data)
        if features is None or len(features) < 4:
            return 0
        try:
            detector = HMMRegimeDetector(n_states=4)
            detector.fit(features)
            states, _ = detector.predict_regime(features)
            return int(states[-1])
        except (ValueError, RuntimeError) as error:
            logger.warning("Regime detection failed; using Emerging: %s", error)
            return 0

    @staticmethod
    def _extract_regime_features(data: pd.DataFrame) -> np.ndarray | None:
        columns = [column for column in ("ev_stock_share", "growth_rate") if column in data.columns]
        if len(columns) < 2:
            return None
        features = data[columns].copy()
        if not features.notna().any().all():
            return None
        return features.apply(lambda series: series.fillna(series.median())).to_numpy(dtype=float)

    def _simulate(
        self,
        M: float,
        p: float,
        q: float,
        t_hist: np.ndarray,
        horizon: int,
        prior: dict,
        last_year: int,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        t_future = np.arange(t_hist[-1] + 1, t_hist[-1] + horizon + 1, dtype=float)
        paths = np.empty((self.n_simulations, horizon), dtype=float)
        q_adjusted = max(q * prior["q_mult"], 0.0)
        low, high = prior["M_range"]
        volatility = float(prior["volatility"])

        for index in range(self.n_simulations):
            market_potential = max(M * rng.uniform(low, high), 1e-8)
            previous = BassDiffusion.cumulative_adoption(t_hist[-1], market_potential, p, q_adjusted)
            future = BassDiffusion.cumulative_adoption(t_future, market_potential, p, q_adjusted)
            annual = np.diff(np.concatenate(([previous], future)))
            process_noise = np.exp(rng.normal(-0.5 * volatility**2, volatility, horizon))
            paths[index] = np.maximum(annual * process_noise, 0.0)

        return pd.DataFrame(
            {
                "year": [last_year + step for step in range(1, horizon + 1)],
                "median": np.percentile(paths, 50, axis=0),
                "p10": np.percentile(paths, 10, axis=0),
                "p90": np.percentile(paths, 90, axis=0),
                "p5": np.percentile(paths, 5, axis=0),
                "p95": np.percentile(paths, 95, axis=0),
                "mean": np.mean(paths, axis=0),
            }
        )

    @staticmethod
    def _build_narrative(regime_id: int, prior: dict, M: float, p: float, q: float, baseline: pd.DataFrame) -> dict:
        regime_name = prior["name"]
        peak_row = baseline.loc[baseline["median"].idxmax()]
        spread_80 = float((baseline["p90"] - baseline["p10"]).mean())
        return {
            "what": f"{regime_name} regime scenario. Bass parameters: M={M:,.0f}, p={p:.4f}, q={q:.4f}.",
            "why": (
                f"The scenario applies q × {prior['q_mult']:.2f}, samples M between "
                f"{prior['M_range'][0]:.0%} and {prior['M_range'][1]:.0%} of the fitted value, "
                "and uses lognormal process noise."
            ),
            "implication": (
                f"Median annual sales peak near {int(peak_row['year'])} at {peak_row['median']:,.0f} units. "
                f"The mean P10–P90 spread is {spread_80:,.0f} units per year."
            ),
            "limitation": "Intervals are empirical scenario quantiles, not confidence or credible intervals.",
        }
