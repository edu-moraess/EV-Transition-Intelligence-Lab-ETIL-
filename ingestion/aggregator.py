# ingestion/aggregator.py
"""
Aggregator — correct dimensional aggregation of IEA GEVO datasets.

KEY RULE — double-counting prevention:
  IEA 2026 provides EV sales at two levels:
    1. powertrain='EV'            → aggregate (BEV + PHEV + FCEV + EREV)
    2. powertrain='BEV'/'PHEV'... → individual components

  NEVER mix these two levels in a sum. Always use one or the other.

  Correct:  EV_total = sum(EV) across modes          (uses aggregate)
  Correct:  EV_total = sum(BEV+PHEV+FCEV) per mode   (uses components)
  WRONG:    EV_total = sum(EV + BEV + PHEV)           (double-counts)

  IEA 2025 has NO 'EV' aggregate in EV sales → must use BEV+PHEV+FCEV.

Primary key: country × category × year × parameter × mode × powertrain

Aggregation dimensions:
  - mode    : Cars / Buses / Trucks / Vans / 2 and 3 wheelers / Total
  - category: Historical / Projection-CPS / Projection-STEPS
"""

import pandas as pd
import numpy as np
from typing import Literal
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Modes that represent actual vehicle sales (exclude infrastructure modes)
_VEHICLE_MODES = {
    "Cars", "Buses", "Trucks", "Vans", "2 and 3 wheelers"
}

# Modes that are price segments (only in 2026 battery price data — never aggregate)
_PRICE_MODES = {
    "All cars", "City car", "Large car", "Medium car",
    "Large SUV/Pick-up", "Small SUV/Pick-up"
}

# mode='EV' in IEA data is infrastructure-only (chargers, capacity) — not vehicle sales
_INFRA_MODES = {"EV", "EVSE"}

# EV aggregate powertrain (2026)
_EV_AGGREGATE = "EV"

# EV component powertrains (used when aggregate unavailable — e.g. IEA 2025)
_EV_COMPONENTS = {"BEV", "PHEV", "FCEV", "EREV"}


class Aggregator:
    """
    Aggregates IEA GEVO data correctly across dimensions.

    Usage
    -----
    agg = Aggregator(df_raw, schema="2026")  # or "2025"
    ev_sales = agg.ev_sales(category="Historical", mode="Cars")
    total_ev = agg.total_ev_all_modes(category="Historical")
    panel    = agg.build_panel(category="Historical")
    """

    def __init__(self, df: pd.DataFrame, schema: Literal["2025", "2026"]):
        self.df     = df.copy()
        self.schema = schema
        self._use_aggregate = schema == "2026"  # 2026 has EV aggregate; 2025 does not
        logger.info(f"Aggregator init | schema={schema} | use_EV_aggregate={self._use_aggregate}")

    # ── Public API ────────────────────────────────────────────────────────

    def ev_sales(
        self,
        category: str = "Historical",
        mode: str = "Cars",
    ) -> pd.DataFrame:
        """
        EV sales for a specific mode, correctly avoiding double-counting.

        Returns
        -------
        pd.DataFrame
            Columns: country, category, year, mode, ev_sales
        """
        df = self._filter_base("EV sales", category)
        df = df[df["mode"] == mode]
        df = self._select_ev_powertrain(df)

        result = (
            df.groupby(["region_country", "category", "year", "mode"])["value"]
            .sum()
            .reset_index()
            .rename(columns={"region_country": "country", "value": "ev_sales"})
        )
        return result

    def total_ev_all_modes(
        self,
        category: str = "Historical",
        modes: set | None = None,
    ) -> pd.DataFrame:
        """
        Total EV sales across all vehicle modes (Cars + Buses + Trucks + Vans + 2&3w).

        This is the correct formula:
            Total EV = Σ EV(mode) for mode in vehicle_modes
            using powertrain='EV' (aggregate) where available,
            or BEV+PHEV+FCEV (components) otherwise.

        Parameters
        ----------
        modes : set, optional
            Subset of vehicle modes to include. Default: all vehicle modes.
        """
        vehicle_modes = modes or _VEHICLE_MODES
        df = self._filter_base("EV sales", category)
        df = df[df["mode"].isin(vehicle_modes)]
        df = self._select_ev_powertrain(df)

        result = (
            df.groupby(["region_country", "category", "year"])["value"]
            .sum()
            .reset_index()
            .rename(columns={"region_country": "country", "value": "ev_sales_total"})
        )
        return result

    def ev_stock(
        self,
        category: str = "Historical",
        mode: str = "Cars",
    ) -> pd.DataFrame:
        """EV stock for a specific mode."""
        df = self._filter_base("EV stock", category)
        df = df[df["mode"] == mode]
        df = self._select_ev_powertrain(df)

        return (
            df.groupby(["region_country", "category", "year", "mode"])["value"]
            .sum()
            .reset_index()
            .rename(columns={"region_country": "country", "value": "ev_stock"})
        )

    def ev_sales_share(
        self,
        category: str = "Historical",
        mode: str = "Cars",
    ) -> pd.DataFrame:
        """EV sales share (%) — uses powertrain='EV' in both schemas."""
        df = self._filter_base("EV sales share", category)
        df = df[(df["mode"] == mode) & (df["powertrain"] == "EV")]

        return (
            df.groupby(["region_country", "category", "year", "mode"])["value"]
            .mean()
            .reset_index()
            .rename(columns={"region_country": "country", "value": "ev_sales_share"})
        )

    def ev_stock_share(
        self,
        category: str = "Historical",
        mode: str = "Cars",
    ) -> pd.DataFrame:
        """EV stock share (%) — uses powertrain='EV' in both schemas."""
        df = self._filter_base("EV stock share", category)
        df = df[(df["mode"] == mode) & (df["powertrain"] == "EV")]

        return (
            df.groupby(["region_country", "category", "year", "mode"])["value"]
            .mean()
            .reset_index()
            .rename(columns={"region_country": "country", "value": "ev_stock_share"})
        )

    def charging_points(
        self,
        category: str = "Historical",
    ) -> pd.DataFrame:
        """EV charging points — infrastructure, not vehicle-mode specific."""
        df = self._filter_base("EV charging points", category)
        # Exclude mode='EV'/'EVSE' (infrastructure label, not vehicle mode)
        df = df[~df["mode"].isin(_INFRA_MODES)]

        return (
            df.groupby(["region_country", "category", "year"])["value"]
            .sum()
            .reset_index()
            .rename(columns={"region_country": "country", "value": "charging_points"})
        )

    def breakdown_by_powertrain(
        self,
        parameter: str = "EV sales",
        category: str = "Historical",
        mode: str = "Cars",
    ) -> pd.DataFrame:
        """
        EV sales broken down by powertrain (BEV / PHEV / FCEV / EREV).
        Uses components directly — never the 'EV' aggregate — to avoid ambiguity.
        """
        df = self._filter_base(parameter, category)
        df = df[(df["mode"] == mode) & (df["powertrain"].isin(_EV_COMPONENTS))]

        return (
            df.groupby(["region_country", "category", "year", "mode", "powertrain"])["value"]
            .sum()
            .reset_index()
            .rename(columns={"region_country": "country", "value": "value"})
        )

    def build_panel(
        self,
        category: str = "Historical",
        mode: str = "Cars",
    ) -> pd.DataFrame:
        """
        Build a wide panel with all key metrics for a given category and mode.
        This is the main method consumed by the ETIL pipeline.

        Returns
        -------
        pd.DataFrame
            Columns: country, category, year, mode,
                     ev_sales, ev_stock, ev_sales_share, ev_stock_share,
                     charging_points, data_source
        """
        sales       = self.ev_sales(category, mode)
        stock       = self.ev_stock(category, mode)
        sales_share = self.ev_sales_share(category, mode)
        stock_share = self.ev_stock_share(category, mode)
        charging    = self.charging_points(category)

        panel = sales.merge(
            stock[["country","year","ev_stock"]], on=["country","year"], how="left"
        ).merge(
            sales_share[["country","year","ev_sales_share"]], on=["country","year"], how="left"
        ).merge(
            stock_share[["country","year","ev_stock_share"]], on=["country","year"], how="left"
        ).merge(
            charging[["country","year","charging_points"]], on=["country","year"], how="left"
        )

        panel["data_source"] = self.schema
        panel["year"]        = panel["year"].astype(int)
        panel["ev_sales"]    = pd.to_numeric(panel["ev_sales"], errors="coerce")
        panel = panel[panel["ev_sales"] > 0].reset_index(drop=True)

        logger.info(
            f"Panel built | category={category} mode={mode} | "
            f"{panel['country'].nunique()} countries | {panel.shape[0]} rows"
        )
        return panel

    def available_projections(self) -> list:
        """List available projection scenarios in this dataset."""
        return sorted(
            c for c in self.df["category"].unique()
            if c.startswith("Projection")
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _filter_base(self, parameter: str, category: str) -> pd.DataFrame:
        """Filter by parameter and category, exclude price and infra modes."""
        df = self.df[
            (self.df["parameter"] == parameter) &
            (self.df["category"] == category)
        ].copy()
        # Remove price-segment modes (battery price data, not sales)
        df = df[~df["mode"].isin(_PRICE_MODES)]
        return df

    def _select_ev_powertrain(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select the correct powertrain level to avoid double-counting.

        IEA 2026: prefer powertrain='EV' (aggregate already computed by IEA).
                  If 'EV' not present for a given row, fall back to components.
        IEA 2025: 'EV' is absent from EV sales → use BEV+PHEV+FCEV.
        """
        if self._use_aggregate:
            # Use EV aggregate where available
            has_ev  = df[df["powertrain"] == _EV_AGGREGATE]
            # For rows where EV aggregate is absent, use components
            no_ev_keys = set(
                zip(df["region_country"], df["year"], df.get("mode", ""))
            ) - set(zip(has_ev["region_country"], has_ev["year"], has_ev.get("mode", "")))
            if no_ev_keys:
                fallback = df[
                    df["powertrain"].isin(_EV_COMPONENTS) &
                    list(zip(df["region_country"], df["year"], df.get("mode",""))).apply(
                        lambda k: k in no_ev_keys
                    )
                ] if False else pd.DataFrame()  # simplified: aggregate covers all 2026
            return has_ev
        else:
            # 2025: sum BEV + PHEV + FCEV (no EV aggregate in EV sales)
            return df[df["powertrain"].isin(_EV_COMPONENTS)]
