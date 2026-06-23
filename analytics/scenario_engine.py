# analytics/scenario_engine.py
"""
Scenario Engine — simulates EV adoption under alternative policy assumptions.

Built on Bass diffusion. Three parameters are adjustable:
  p — innovation coefficient (early adopter / policy pull)
  q — imitation coefficient (word of mouth / network effects)
  M — market ceiling (infrastructure, income, EV model availability)

param_adjust format: {"p": 0.30, "q": 0.20, "M": 0.40}
  → positive values increase the parameter by that fraction
  → negative values decrease it
  → empty dict = baseline (no change)
"""
import pandas as pd
import numpy as np
from models.diffusion import BassDiffusion
from typing import Dict
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ScenarioEngine:

    def __init__(self, df: pd.DataFrame, country: str):
        self.country = country
        self.hist = (
            df[df["country"] == country]
            .sort_values("year")
            .reset_index(drop=True)
        )
        if self.hist.empty:
            raise ValueError(f"No data for country: {country}")

    def run_scenario(
        self,
        scenario_name: str,
        param_adjust: Dict[str, float],
        horizon: int = 10,
    ) -> pd.DataFrame:
        """
        Fit Bass on historical data, apply parameter adjustments, project forward.

        Returns
        -------
        pd.DataFrame
            columns: year, scenario, annual_sales, cumulative_sales
            Includes both historical (for plotting continuity) and forecast rows.
        """
        t = self.hist.index.values.astype(float)
        S = self.hist["ev_sales"].cumsum().values

        # Fit baseline
        bass = BassDiffusion()
        try:
            M, p, q = bass.fit(t, S)
        except Exception as e:
            logger.warning(f"Bass fit failed for {self.country}: {e}")
            return pd.DataFrame()

        # Apply adjustments (additive fraction)
        p = max(p * (1 + param_adjust.get("p", 0)), 1e-6)
        q = max(q * (1 + param_adjust.get("q", 0)), 1e-6)
        M = max(M * (1 + param_adjust.get("M", 0)), S[-1] * 1.01)

        # Forecast
        t_all = np.arange(0, len(t) + horizon, dtype=float)
        S_all = np.array([bass.cumulative_adoption(ti, M, p, q) for ti in t_all])

        # Annual sales = first diff of cumulative
        ann_all = np.diff(S_all, prepend=0)
        ann_all = np.maximum(ann_all, 0)  # no negative sales

        last_hist_year = int(self.hist["year"].iloc[-1])
        years_hist     = self.hist["year"].values
        years_future   = [last_hist_year + i + 1 for i in range(horizon)]
        all_years      = list(years_hist) + years_future

        return pd.DataFrame({
            "year":             all_years,
            "scenario":         scenario_name,
            "annual_sales":     ann_all[:len(all_years)],
            "cumulative_sales": S_all[:len(all_years)],
            "is_forecast":      [False] * len(years_hist) + [True] * horizon,
        })

    def run_all(
        self,
        optimistic_adj: Dict[str, float],
        conservative_adj: Dict[str, float],
        horizon: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        """Convenience method: run baseline + optimistic + conservative."""
        return {
            "Baseline":     self.run_scenario("Baseline",     {},                horizon),
            "Optimistic":   self.run_scenario("Optimistic",   optimistic_adj,    horizon),
            "Conservative": self.run_scenario("Conservative", conservative_adj,  horizon),
        }
