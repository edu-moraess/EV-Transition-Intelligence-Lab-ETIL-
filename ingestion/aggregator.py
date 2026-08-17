"""Dimensionally safe aggregation for IEA Global EV Outlook datasets."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)

_VEHICLE_MODES = {"Cars", "Buses", "Trucks", "Vans", "2 and 3 wheelers"}
_PRICE_MODES = {
    "All cars",
    "City car",
    "Large car",
    "Medium car",
    "Large SUV/Pick-up",
    "Small SUV/Pick-up",
}
_INFRA_MODES = {"EV", "EVSE"}
_EV_AGGREGATE = "EV"
_EV_COMPONENTS = {"BEV", "PHEV", "FCEV", "EREV"}


class Aggregator:
    """Aggregate IEA records while preserving their dimensional meaning."""

    def __init__(self, df: pd.DataFrame, schema: Literal["2025", "2026"]):
        required = {"region_country", "category", "year", "parameter", "mode", "powertrain", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for aggregation: {sorted(missing)}")
        self.df = df.copy()
        self.schema = schema
        self._use_aggregate = schema == "2026"

    def ev_sales(self, category: str = "Historical", mode: str = "Cars") -> pd.DataFrame:
        """Return EV sales for one vehicle mode."""
        return self._aggregate_vehicle_metric("EV sales", "ev_sales", category, mode)

    def ev_stock(self, category: str = "Historical", mode: str = "Cars") -> pd.DataFrame:
        """Return EV stock for one vehicle mode."""
        return self._aggregate_vehicle_metric("EV stock", "ev_stock", category, mode)

    def total_ev_all_modes(
        self,
        category: str = "Historical",
        modes: set[str] | None = None,
    ) -> pd.DataFrame:
        """Return EV sales summed over vehicle modes without double counting."""
        return self._aggregate_all_vehicle_modes("EV sales", "ev_sales_total", category, modes)

    def total_ev_stock_all_modes(
        self,
        category: str = "Historical",
        modes: set[str] | None = None,
    ) -> pd.DataFrame:
        """Return EV stock summed over vehicle modes without double counting."""
        return self._aggregate_all_vehicle_modes("EV stock", "ev_stock_total", category, modes)

    def ev_sales_share(self, category: str = "Historical", mode: str = "Cars") -> pd.DataFrame:
        """Return the official EV sales share in percent for one vehicle mode."""
        return self._aggregate_share("EV sales share", "ev_sales_share", category, mode)

    def ev_stock_share(self, category: str = "Historical", mode: str = "Cars") -> pd.DataFrame:
        """Return the official EV stock share in percent for one vehicle mode."""
        return self._aggregate_share("EV stock share", "ev_stock_share", category, mode)

    def total_ev_sales_share(
        self,
        category: str = "Historical",
        modes: set[str] | None = None,
    ) -> pd.DataFrame:
        """Return a denominator-weighted EV sales share across vehicle modes.

        The IEA publishes mode-specific EV shares but not a single all-mode
        denominator in the local extracts. For each mode, total sales are
        reconstructed as ``EV sales / share`` and then summed. The total share
        is returned only when every mode with EV sales has a valid share.
        """
        return self._aggregate_total_share(
            value_parameter="EV sales",
            share_parameter="EV sales share",
            value_name="ev_sales_share",
            category=category,
            modes=modes,
        )

    def total_ev_stock_share(
        self,
        category: str = "Historical",
        modes: set[str] | None = None,
    ) -> pd.DataFrame:
        """Return a denominator-weighted EV stock share across vehicle modes."""
        return self._aggregate_total_share(
            value_parameter="EV stock",
            share_parameter="EV stock share",
            value_name="ev_stock_share",
            category=category,
            modes=modes,
        )

    def charging_points(self, category: str = "Historical") -> pd.DataFrame:
        """Return public charging points across the IEA 2025 and 2026 schemas."""
        df = self.df[
            (self.df["parameter"] == "EV charging points")
            & (self.df["category"] == category)
            & (self.df["mode"].isin(_INFRA_MODES))
        ]
        return self._sum(df, ["region_country", "category", "year"], "charging_points")

    def charging_by_type(self, category: str = "Historical") -> pd.DataFrame:
        """Return charging points by charger type."""
        df = self.df[
            (self.df["parameter"] == "EV charging points")
            & (self.df["category"] == category)
            & (self.df["mode"].isin(_INFRA_MODES))
        ]
        result = self._sum(df, ["region_country", "category", "year", "powertrain"], "charging_points")
        return result.rename(columns={"powertrain": "charger_type"})

    def ev_per_charger(self, category: str = "Historical") -> pd.DataFrame:
        """Return the IEA-provided electric LDV-per-charger ratio."""
        df = self.df[
            (self.df["parameter"] == "Electric LDVs per charging point")
            & (self.df["category"] == category)
        ]
        return self._mean(df, ["region_country", "category", "year"], "ev_per_charger")

    def breakdown_by_powertrain(
        self,
        parameter: str = "EV sales",
        category: str = "Historical",
        mode: str = "Cars",
    ) -> pd.DataFrame:
        """Return component-level EV values without the aggregate EV record."""
        df = self._filter_base(parameter, category)
        df = df[(df["mode"] == mode) & (df["powertrain"].isin(_EV_COMPONENTS))]
        result = self._sum(
            df,
            ["region_country", "category", "year", "mode", "powertrain"],
            "value",
        )
        return result

    def build_panel(self, category: str = "Historical", mode: str = "Cars") -> pd.DataFrame:
        """Build the canonical country-year panel for a single vehicle mode."""
        sales = self.ev_sales(category, mode)
        stock = self.ev_stock(category, mode)
        sales_share = self.ev_sales_share(category, mode)
        stock_share = self.ev_stock_share(category, mode)
        charging = self.charging_points(category)
        panel = (
            sales.merge(stock[["country", "year", "ev_stock"]], on=["country", "year"], how="left")
            .merge(sales_share[["country", "year", "ev_sales_share"]], on=["country", "year"], how="left")
            .merge(stock_share[["country", "year", "ev_stock_share"]], on=["country", "year"], how="left")
            .merge(charging[["country", "year", "charging_points"]], on=["country", "year"], how="left")
        )
        panel["data_source"] = self.schema
        panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
        panel["ev_sales"] = pd.to_numeric(panel["ev_sales"], errors="coerce")
        panel = panel.loc[panel["ev_sales"] > 0].drop_duplicates(["country", "year"]).reset_index(drop=True)
        return panel

    def available_projections(self) -> list[str]:
        """Return available IEA projection scenario names."""
        return sorted(category for category in self.df["category"].dropna().unique() if category.startswith("Projection"))

    def _aggregate_vehicle_metric(self, parameter: str, value_name: str, category: str, mode: str) -> pd.DataFrame:
        df = self._filter_base(parameter, category)
        df = self._select_ev_powertrain(df[df["mode"] == mode])
        return self._sum(df, ["region_country", "category", "year", "mode"], value_name)

    def _aggregate_all_vehicle_modes(
        self,
        parameter: str,
        value_name: str,
        category: str,
        modes: set[str] | None,
    ) -> pd.DataFrame:
        df = self._filter_base(parameter, category)
        df = self._select_ev_powertrain(df[df["mode"].isin(modes or _VEHICLE_MODES)])
        return self._sum(df, ["region_country", "category", "year"], value_name)

    def _aggregate_share(self, parameter: str, value_name: str, category: str, mode: str) -> pd.DataFrame:
        df = self._filter_base(parameter, category)
        df = df[(df["mode"] == mode) & (df["powertrain"] == _EV_AGGREGATE)]
        return self._mean(df, ["region_country", "category", "year", "mode"], value_name)

    def _aggregate_total_share(
        self,
        value_parameter: str,
        share_parameter: str,
        value_name: str,
        category: str,
        modes: set[str] | None,
    ) -> pd.DataFrame:
        vehicle_modes = modes or _VEHICLE_MODES
        value_df = self._filter_base(value_parameter, category)
        value_df = self._select_ev_powertrain(value_df[value_df["mode"].isin(vehicle_modes)])
        value_keys = ["region_country", "category", "year", "mode"]
        values = self._sum(value_df, value_keys, "numerator")

        share_df = self._filter_base(share_parameter, category)
        share_df = share_df[
            share_df["mode"].isin(vehicle_modes)
            & share_df["powertrain"].eq(_EV_AGGREGATE)
        ]
        shares = self._mean(share_df, value_keys, "share")
        merged = values.merge(shares, on=["country", "category", "year", "mode"], how="left")
        merged["denominator"] = np.where(
            merged["share"].gt(0),
            merged["numerator"] / (merged["share"] / 100.0),
            np.nan,
        )

        grouped = (
            merged.groupby(["country", "category", "year"], dropna=False)
            .agg(
                numerator=("numerator", "sum"),
                denominator=("denominator", "sum"),
                modes_total=("mode", "count"),
                modes_with_share=("denominator", lambda values: values.notna().sum()),
            )
            .reset_index()
        )
        grouped[value_name] = np.where(
            grouped["modes_total"].eq(grouped["modes_with_share"])
            & grouped["denominator"].gt(0),
            grouped["numerator"] / grouped["denominator"] * 100.0,
            np.nan,
        )
        grouped["share_coverage"] = np.divide(
            grouped["modes_with_share"],
            grouped["modes_total"],
            out=np.zeros(len(grouped), dtype=float),
            where=grouped["modes_total"].gt(0),
        )
        return grouped[["country", "category", "year", value_name, "share_coverage"]]

    def _filter_base(self, parameter: str, category: str) -> pd.DataFrame:
        df = self.df[(self.df["parameter"] == parameter) & (self.df["category"] == category)].copy()
        return df.loc[~df["mode"].isin(_PRICE_MODES)]

    def _select_ev_powertrain(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._use_aggregate:
            return df.loc[df["powertrain"].isin(_EV_COMPONENTS)].copy()
        key_columns = ["region_country", "category", "year", "mode"]
        aggregate = df.loc[df["powertrain"] == _EV_AGGREGATE].copy()
        components = df.loc[df["powertrain"].isin(_EV_COMPONENTS)].copy()
        if aggregate.empty:
            return components
        if components.empty:
            return aggregate
        aggregate_keys = pd.MultiIndex.from_frame(aggregate[key_columns])
        component_keys = pd.MultiIndex.from_frame(components[key_columns])
        fallback = components.loc[~component_keys.isin(aggregate_keys)]
        return pd.concat([aggregate, fallback], ignore_index=True)

    @staticmethod
    def _sum(df: pd.DataFrame, keys: list[str], value_name: str) -> pd.DataFrame:
        result = df.groupby(keys, dropna=False)["value"].sum(min_count=1).reset_index()
        return result.rename(columns={"region_country": "country", "value": value_name})

    @staticmethod
    def _mean(df: pd.DataFrame, keys: list[str], value_name: str) -> pd.DataFrame:
        result = df.groupby(keys, dropna=False)["value"].mean().reset_index()
        return result.rename(columns={"region_country": "country", "value": value_name})
