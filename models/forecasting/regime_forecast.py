# models/forecasting/regime_forecast.py
"""
RegimeAwareForecastEngine

Conditions the Bass diffusion forecast on the current adoption regime,
producing regime-specific trajectories and confidence intervals.

Four regimes produce different forecast priors:

  Emerging      (0): low p, low q, high uncertainty — anything is possible
  Expansion     (1): moderate p+q, medium uncertainty — trend is establishing
  Acceleration  (2): high q (imitation dominant), low uncertainty — trend is clear
  Maturity      (3): low growth headroom, high certainty — saturation approaching

These translate into three concrete modifications to the Bass model:
  1. Market ceiling M: scaled by regime (Emerging = wider range, Maturity = tighter)
  2. Growth coefficient: adjusted q by regime
  3. Monte Carlo volatility: higher in earlier regimes, lower in later ones

Anti-leakage: regime is determined from data up to year t, forecast from t+1 onward.
"""

import numpy as np
import pandas as pd
from typing import Optional
from models.diffusion.bass import BassDiffusion
from models.regimes.hmm import HMMRegimeDetector
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Regime-specific priors (multipliers applied to Bass fitted parameters)
_REGIME_PRIORS = {
    0: {"M_range": (0.7, 2.5), "q_mult": 0.8,  "volatility": 0.45, "name": "Emerging"},
    1: {"M_range": (0.8, 1.8), "q_mult": 1.0,  "volatility": 0.30, "name": "Expansion"},
    2: {"M_range": (0.9, 1.4), "q_mult": 1.25, "volatility": 0.18, "name": "Acceleration"},
    3: {"M_range": (0.95, 1.1),"q_mult": 0.9,  "volatility": 0.10, "name": "Maturity"},
}

_N_SIMULATIONS = 2000
_CI_LEVELS     = [0.10, 0.20]   # 80% and 95% CI bands


class RegimeAwareForecastEngine:
    """
    Produces regime-conditioned Bass diffusion forecasts.

    Usage
    -----
    engine = RegimeAwareForecastEngine()
    result = engine.forecast(country_df, horizon=10)

    result keys:
      - regime_id         current regime
      - regime_name       human-readable name
      - baseline          pd.DataFrame (year, median, p10, p90, p5, p95, annual_sales_*)
      - all_regimes       dict of DataFrames, one per regime, for comparison
      - bass_params       dict (M, p, q) fitted on historical data
      - narrative         dict with what/why/implication strings
    """

    def __init__(self, n_simulations: int = _N_SIMULATIONS):
        self.n_simulations = n_simulations

    def forecast(
        self,
        country_df: pd.DataFrame,
        horizon: int = 10,
        regime_id: Optional[int] = None,
    ) -> dict:
        """
        Regime-aware forecast for a single country.

        Parameters
        ----------
        country_df : country panel, sorted by year
        horizon    : forecast years beyond last data point
        regime_id  : override auto-detected regime (useful for what-if analysis)

        Returns
        -------
        dict with regime metadata, forecast DataFrames, narrative
        """
        if len(country_df) < 4:
            return {"error": "Insufficient data (need at least 4 years)"}

        data = country_df.sort_values("year").reset_index(drop=True)

        # ── Detect current regime ──────────────────────────────────────────────
        if regime_id is None:
            regime_id = self._detect_regime(data)
        regime_name = _REGIME_PRIORS.get(regime_id, {}).get("name", f"Regime {regime_id}")

        # ── Fit Bass on historical data ────────────────────────────────────────
        t = np.arange(len(data), dtype=float)
        S = data["ev_sales"].cumsum().values
        try:
            bass = BassDiffusion()
            M, p, q = bass.fit(t, S)
        except Exception as e:
            return {"error": f"Bass fit failed: {e}"}

        prior = _REGIME_PRIORS[regime_id]

        # ── Baseline forecast (regime-adjusted) ───────────────────────────────
        baseline = self._simulate(M, p, q, t, horizon, prior, data["year"].iloc[-1])

        # ── All-regime comparison (for "what-if different regime?") ───────────
        all_regimes = {}
        for rid, rprior in _REGIME_PRIORS.items():
            all_regimes[rid] = self._simulate(M, p, q, t, horizon, rprior, data["year"].iloc[-1])

        # ── Narrative ─────────────────────────────────────────────────────────
        narrative = self._build_narrative(
            regime_id, regime_name, prior, M, p, q, baseline, data
        )

        return {
            "regime_id":    regime_id,
            "regime_name":  regime_name,
            "baseline":     baseline,
            "all_regimes":  all_regimes,
            "bass_params":  {"M": M, "p": p, "q": q},
            "prior_applied": prior,
            "narrative":    narrative,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _detect_regime(self, data: pd.DataFrame) -> int:
        """Detect current regime from the last 3-5 data points."""
        try:
            features = self._extract_regime_features(data)
            if features is None or len(features) < 2:
                return 0  # default to Emerging

            detector = HMMRegimeDetector(n_states=4)
            detector.fit(features)
            states, _ = detector.predict_regime(features)
            return int(states[-1])  # current year regime
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}. Defaulting to Emerging.")
            return 0

    def _extract_regime_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for regime detection."""
        feats = []
        for col in ["ev_stock_share", "growth_rate"]:
            if col in data.columns and data[col].notna().any():
                feats.append(data[col].fillna(data[col].median()).values)
        if len(feats) < 2:
            return None
        return np.column_stack(feats)

    def _simulate(
        self,
        M: float, p: float, q: float,
        t_hist: np.ndarray,
        horizon: int,
        prior: dict,
        last_year: int,
    ) -> pd.DataFrame:
        """
        Monte Carlo simulation with regime-specific priors.

        Samples M from the regime's M_range multiplier,
        applies q multiplier, adds stochastic noise scaled by volatility.
        """
        n_hist   = len(t_hist)
        t_future = np.arange(n_hist, n_hist + horizon, dtype=float)
        all_years = [last_year + i + 1 for i in range(horizon)]

        paths = np.zeros((self.n_simulations, horizon))
        rng = np.random.default_rng(42)

        q_adj = q * prior["q_mult"]
        vol   = prior["volatility"]
        M_lo, M_hi = prior["M_range"]

        for i in range(self.n_simulations):
            M_sim = M * rng.uniform(M_lo, M_hi)
            noise = rng.normal(1.0, vol, horizon)
            try:
                bass_sim = BassDiffusion()
                cum_sim  = np.array([
                    bass_sim.cumulative_adoption(ti, M_sim, p, q_adj)
                    for ti in t_future
                ])
                ann_sim  = np.maximum(np.diff(cum_sim, prepend=cum_sim[0]), 0)
                paths[i] = ann_sim * noise
            except Exception:
                paths[i] = np.full(horizon, np.nan)

        valid = paths[~np.isnan(paths).any(axis=1)]
        if len(valid) == 0:
            valid = paths

        return pd.DataFrame({
            "year":   all_years,
            "median": np.percentile(valid, 50, axis=0),
            "p10":    np.percentile(valid, 10, axis=0),
            "p90":    np.percentile(valid, 90, axis=0),
            "p5":     np.percentile(valid,  5, axis=0),
            "p95":    np.percentile(valid, 95, axis=0),
            "mean":   np.mean(valid, axis=0),
        })

    def _build_narrative(
        self,
        regime_id: int, regime_name: str, prior: dict,
        M: float, p: float, q: float,
        baseline: pd.DataFrame,
        data: pd.DataFrame,
    ) -> dict:
        """Build structured narrative for the forecast."""
        vol_desc = {0: "high", 1: "moderate", 2: "low", 3: "very low"}[regime_id]
        peak_year = int(baseline.loc[baseline["median"].idxmax(), "year"])
        peak_val  = baseline["median"].max()
        spread_80 = float((baseline["p90"] - baseline["p10"]).mean())

        what = (
            f"{regime_name} regime detected. "
            f"Bass parameters: M={M:,.0f}, p={p:.4f}, q={q:.4f}."
        )
        why = (
            f"In the {regime_name} phase, imitation coefficient is adjusted by "
            f"{(prior['q_mult']-1)*100:+.0f}% and forecast uncertainty is {vol_desc}. "
            f"Market ceiling scaled to [{prior['M_range'][0]:.0%}–{prior['M_range'][1]:.0%}] of fitted M."
        )
        implication = (
            f"Peak annual sales projected around {peak_year} "
            f"(median {peak_val/1e3:.0f}k units). "
            f"80% CI spread averages {spread_80/1e3:.0f}k units/year — "
            f"{'high uncertainty, wide range of outcomes' if spread_80 > 50000 else 'relatively contained uncertainty'}."
        )

        return {"what": what, "why": why, "implication": implication}
