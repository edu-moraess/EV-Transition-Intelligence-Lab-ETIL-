"""Compatibility facade for the canonical IEA data loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ingestion.loader import IEALoader


class DatasetEngine:
    """Expose the legacy application interface using the canonical IEA loader."""

    def __init__(self, raw_dir: Path | None = None):
        self.loader = IEALoader(raw_dir=raw_dir)

    def load_dataset(
        self,
        year: Literal[2025, 2026],
        vehicle_mode: str = "Cars",
        countries_only: bool = True,
    ) -> pd.DataFrame:
        """Load a historical, country-level panel for one IEA release."""
        if vehicle_mode == "Total":
            return self._load_total(year, countries_only)
        return self.loader.load_dataset(
            year,
            category="Historical",
            mode=vehicle_mode,
            countries_only=countries_only,
        )

    def load_all_datasets(
        self,
        vehicle_mode: str = "Cars",
        countries_only: bool = True,
    ) -> pd.DataFrame:
        """Load the 2025 and 2026 panels with 2026 priority on overlapping keys."""
        if vehicle_mode == "Total":
            panels = [self._load_total(year, countries_only) for year in (2025, 2026)]
            combined = pd.concat(panels, ignore_index=True)
            combined = combined.sort_values("data_source", ascending=False)
            combined = combined.drop_duplicates(["country", "year"], keep="first")
            combined["data_source"] = "combined"
            return combined.sort_values(["country", "year"]).reset_index(drop=True)
        return self.loader.load_all_datasets(
            category="Historical",
            mode=vehicle_mode,
            countries_only=countries_only,
        )

    @staticmethod
    def filter_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Return a detached panel slice for the requested year."""
        return df.loc[df["year"] == year].copy()

    def _load_total(self, year: Literal[2025, 2026], countries_only: bool) -> pd.DataFrame:
        aggregator = self.loader.get_aggregator(year, countries_only=countries_only)
        sales = aggregator.total_ev_all_modes(category="Historical")
        stock = aggregator.total_ev_stock_all_modes(category="Historical")
        charging = aggregator.charging_points(category="Historical")
        panel = (
            sales.merge(stock[["country", "year", "ev_stock_total"]], on=["country", "year"], how="left")
            .merge(charging[["country", "year", "charging_points"]], on=["country", "year"], how="left")
            .rename(columns={"ev_sales_total": "ev_sales", "ev_stock_total": "ev_stock"})
        )
        panel["mode"] = "Total"
        panel["ev_sales_share"] = np.nan
        panel["ev_stock_share"] = np.nan
        panel["data_source"] = str(year)
        return panel.sort_values(["country", "year"]).reset_index(drop=True)


__all__ = ["DatasetEngine"]
