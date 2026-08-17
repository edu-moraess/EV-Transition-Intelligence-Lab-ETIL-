"""Compatibility facade for the canonical IEA data loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
            return self.loader.load_total_dataset(
                year,
                category="Historical",
                countries_only=countries_only,
            )
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
            return self.loader.load_all_datasets(
                category="Historical",
                mode="Total",
                countries_only=countries_only,
            )
        return self.loader.load_all_datasets(
            category="Historical",
            mode=vehicle_mode,
            countries_only=countries_only,
        )

    @staticmethod
    def filter_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Return a detached panel slice for the requested year."""
        return df.loc[df["year"] == year].copy()


__all__ = ["DatasetEngine"]
