"""Validation-oriented cleaning for country-year EV panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataCleaner:
    """Apply deterministic, non-imputing cleaning to an ETIL panel."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean(self) -> pd.DataFrame:
        """Validate keys and types while preserving missing observations."""
        self._validate_required_columns()
        self._standardize_countries()
        self._convert_types()
        self._remove_duplicate_keys()
        self._validate_temporal()
        return self.df.sort_values(["country", "year"]).reset_index(drop=True)

    def _validate_required_columns(self) -> None:
        required = {"country", "year", "ev_sales"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required panel columns: {sorted(missing)}")

    def _remove_duplicate_keys(self) -> None:
        duplicate_count = int(self.df.duplicated(subset=["country", "year"], keep=False).sum())
        if duplicate_count:
            raise ValueError(f"Panel contains {duplicate_count} duplicate country-year rows.")

    def _standardize_countries(self) -> None:
        country = self.df["country"].astype("string").str.strip()
        if country.isna().any() or (country == "").any():
            raise ValueError("Panel contains missing or blank country identifiers.")
        self.df["country"] = country

    def _convert_types(self) -> None:
        year = pd.to_numeric(self.df["year"], errors="coerce")
        if year.isna().any() or (year % 1 != 0).any():
            raise ValueError("Panel contains invalid year values.")
        self.df["year"] = year.astype(int)
        numeric_columns = {
            "ev_sales",
            "ev_stock",
            "ev_sales_share",
            "ev_stock_share",
            "charging_points",
            "population",
            "gdp",
            "total_sales",
        }
        for column in numeric_columns & set(self.df.columns):
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce")
        if (self.df["ev_sales"] < 0).any():
            raise ValueError("EV sales must be non-negative.")

    def _validate_temporal(self) -> None:
        for country, years in self.df.groupby("country")["year"]:
            gaps = np.diff(np.sort(years.to_numpy()))
            if (gaps > 1).any():
                logger.warning("Temporal gaps preserved for %s: %s", country, years.tolist())
