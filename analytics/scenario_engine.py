"""Scenario analysis based on constrained Bass diffusion curves."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.diffusion import BassDiffusion
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ScenarioEngine:
    """Simulate alternative Bass-parameter scenarios for one country."""

    def __init__(self, df: pd.DataFrame, country: str):
        self.country = country
        self.hist = df.loc[df["country"] == country].sort_values("year").reset_index(drop=True)
        if self.hist.empty:
            raise ValueError(f"No data for country: {country}")

    def run_scenario(
        self,
        scenario_name: str,
        param_adjust: dict[str, float],
        horizon: int = 10,
    ) -> pd.DataFrame:
        """Fit Bass, apply bounded parameter shocks and return annual and cumulative paths."""
        if horizon < 1:
            raise ValueError("Forecast horizon must be positive.")
        t_hist = np.arange(1, len(self.hist) + 1, dtype=float)
        cumulative_hist = self.hist["ev_sales"].cumsum().to_numpy(dtype=float)
        bass = BassDiffusion()
        try:
            M, p, q = bass.fit(t_hist, cumulative_hist)
        except (RuntimeError, ValueError) as error:
            logger.warning("Bass fit failed for %s: %s", self.country, error)
            return pd.DataFrame()

        p = max(p * (1 + float(param_adjust.get("p", 0))), 1e-8)
        q = max(q * (1 + float(param_adjust.get("q", 0))), 0.0)
        M = max(M * (1 + float(param_adjust.get("M", 0))), cumulative_hist[-1] * (1 + 1e-6))
        t_all = np.arange(1, len(t_hist) + horizon + 1, dtype=float)
        cumulative = bass.cumulative_adoption(t_all, M, p, q)
        annual = np.diff(cumulative, prepend=0.0)

        last_year = int(self.hist["year"].iloc[-1])
        years = list(self.hist["year"]) + [last_year + offset for offset in range(1, horizon + 1)]
        return pd.DataFrame(
            {
                "year": years,
                "scenario": scenario_name,
                "annual_sales": np.maximum(annual, 0.0),
                "cumulative_sales": cumulative,
                "is_forecast": [False] * len(self.hist) + [True] * horizon,
            }
        )

    def run_all(
        self,
        optimistic_adj: dict[str, float],
        conservative_adj: dict[str, float],
        horizon: int = 10,
    ) -> dict[str, pd.DataFrame]:
        """Return baseline, optimistic and conservative scenario paths."""
        return {
            "Baseline": self.run_scenario("Baseline", {}, horizon),
            "Optimistic": self.run_scenario("Optimistic", optimistic_adj, horizon),
            "Conservative": self.run_scenario("Conservative", conservative_adj, horizon),
        }
