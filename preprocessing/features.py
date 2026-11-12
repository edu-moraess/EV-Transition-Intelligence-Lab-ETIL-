"""Feature engineering for chronologically ordered EV country panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.helpers import rolling_momentum
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """Create descriptive features without using future observations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def engineer(self) -> pd.DataFrame:
        """Return a country-year panel with causal feature construction."""
        self._validate_input()
        self.df = self.df.sort_values(["country", "year"]).reset_index(drop=True)
        self._add_penetration()
        self._add_growth_rates()
        self._add_cagr()
        self._add_momentum()
        self._add_volatility()
        self._add_infrastructure_pressure()
        self._add_ev_per_capita()
        return self.df

    def _validate_input(self) -> None:
        required = {"country", "year", "ev_sales"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required feature columns: {sorted(missing)}")
        self.df["year"] = pd.to_numeric(self.df["year"], errors="coerce")
        if self.df["year"].isna().any():
            raise ValueError("Feature engineering requires valid calendar years.")
        if (self.df["year"] % 1 != 0).any():
            raise ValueError("Feature engineering requires integer calendar years.")
        self.df["year"] = self.df["year"].astype(int)
        if self.df.duplicated(["country", "year"]).any():
            raise ValueError("Feature engineering requires unique country-year rows.")

    def _add_penetration(self) -> None:
        if "ev_stock_share" in self.df.columns and self.df["ev_stock_share"].notna().any():
            self.df["penetration"] = self.df["ev_stock_share"]
        elif "total_sales" in self.df.columns and self.df["total_sales"].notna().any():
            total_sales = self.df["total_sales"].replace(0, np.nan)
            self.df["penetration"] = self.df["ev_sales"] / total_sales * 100
        else:
            self.df["penetration"] = np.nan

    def _add_growth_rates(self) -> None:
        """Compute YoY growth only when the immediately preceding calendar year exists."""
        result = pd.Series(np.nan, index=self.df.index, dtype=float)
        for _, group in self.df.groupby("country", sort=False):
            ordered = group.sort_values("year")
            years = ordered["year"].astype(int)
            sales = pd.Series(
                pd.to_numeric(ordered["ev_sales"], errors="coerce").to_numpy(),
                index=years.to_numpy(),
            )
            full_years = range(int(years.min()), int(years.max()) + 1)
            annual_sales = sales.reindex(full_years)
            annual_growth = (
                annual_sales.pct_change(fill_method=None)
                .mul(100)
                .replace([np.inf, -np.inf], np.nan)
            )
            result.loc[ordered.index] = annual_growth.reindex(years.to_numpy()).to_numpy()
        self.df["growth_rate"] = result

    def _add_cagr(self) -> None:
        result = pd.Series(np.nan, index=self.df.index, dtype=float)
        for _, group in self.df.groupby("country", sort=False):
            sales = pd.to_numeric(group["ev_sales"], errors="coerce")
            valid = sales.gt(0) & group["year"].notna()
            if valid.sum() < 2:
                continue
            first_index = valid[valid].index[0]
            initial_sales = sales.loc[first_index]
            initial_year = self.df.loc[first_index, "year"]
            elapsed = self.df.loc[group.index, "year"] - initial_year
            current_sales = sales.loc[group.index]
            valid_elapsed = elapsed.gt(0) & current_sales.gt(0)
            result.loc[group.index[valid_elapsed]] = (
                (current_sales.loc[valid_elapsed] / initial_sales) ** (1 / elapsed.loc[valid_elapsed]) - 1
            ) * 100
        self.df["cagr"] = result

    def _add_momentum(self) -> None:
        """Compute three-year momentum on a complete calendar index.

        Missing calendar years remain NaN in the expanded series, so rolling
        windows spanning a gap are not treated as consecutive annual data.
        """
        result = pd.Series(np.nan, index=self.df.index, dtype=float)
        for _, group in self.df.groupby("country", sort=False):
            ordered = group.sort_values("year")
            years = ordered["year"].astype(int)
            sales = pd.Series(
                pd.to_numeric(ordered["ev_sales"], errors="coerce").to_numpy(),
                index=years.to_numpy(),
            )
            full_years = range(int(years.min()), int(years.max()) + 1)
            annual_sales = sales.reindex(full_years)
            annual_momentum = rolling_momentum(annual_sales, window=3)
            result.loc[ordered.index] = annual_momentum.reindex(years.to_numpy()).to_numpy()
        self.df["momentum_3y"] = result

    def _add_volatility(self) -> None:
        self.df["volatility"] = self.df.groupby("country", group_keys=False)["growth_rate"].apply(
            lambda series: series.rolling(4, min_periods=2).std()
        )

    def _add_infrastructure_pressure(self) -> None:
        if {"ev_stock", "charging_points"}.issubset(self.df.columns):
            chargers = self.df["charging_points"].replace(0, np.nan)
            self.df["ev_per_charger"] = self.df["ev_stock"] / chargers
        else:
            self.df["ev_per_charger"] = np.nan

    def _add_ev_per_capita(self) -> None:
        numerator = "ev_stock" if "ev_stock" in self.df.columns else "ev_sales"
        if "population" in self.df.columns and self.df["population"].notna().any():
            population = self.df["population"].replace(0, np.nan)
            self.df["ev_per_capita"] = self.df[numerator] / population
        else:
            self.df["ev_per_capita"] = np.nan
