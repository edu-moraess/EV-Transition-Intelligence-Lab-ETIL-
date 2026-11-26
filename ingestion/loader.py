# ingestion/loader.py
"""
Loader — canonical entry point for IEA GEVO datasets.

Responsibilities:
  - Load IEA 2025 and/or IEA 2026 raw CSVs
  - Detect schema automatically (2025 vs 2026)
  - Validate column types and dimensional integrity
  - Expose load_dataset() / load_all_datasets() / load_raw()
  - Delegate aggregation to Aggregator

Primary key enforced: country × category × year × parameter × mode × powertrain

Schema differences handled transparently:
  IEA 2026: powertrain='EV' exists in EV sales (aggregate)
  IEA 2025: powertrain='EV' absent from EV sales; only BEV/PHEV/FCEV
"""

from pathlib import Path
from typing import Literal

import pandas as pd

from config.settings import RAW_DATA_DIR
from ingestion.aggregator import Aggregator
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ── File paths ────────────────────────────────────────────────────────────────

_RAW_DIR = RAW_DATA_DIR

_FILES = {
    2025: "EV_Data_Explorer_2025_GEVO_clean.csv",
    2026: "EV_Data_Explorer_2026_GEVO_clean.csv",
}

# ── Valid dimension values ────────────────────────────────────────────────────

_VALID_CATEGORIES  = {"Historical", "Projection-CPS", "Projection-STEPS"}
_VALID_MODES       = {
    "Cars", "Buses", "Trucks", "Vans",
    "2 and 3 wheelers",
    "EV", "EVSE",                          # infrastructure labels
    "All cars", "City car", "Large car",   # battery price segments (2026 only)
    "Medium car", "Large SUV/Pick-up", "Small SUV/Pick-up",
}
_VALID_POWERTRAINS = {
    "EV", "BEV", "PHEV", "FCEV", "EREV", "ICEV", "ALL",
    "Publicly available fast", "Publicly available slow",
    "Publicly available ultra-fast",
}

# Projection years (excluded from Historical panels)
_PROJECTION_YEARS = {2030, 2035, 2040, 2045, 2050}

# Aggregate regions — normalizado em lowercase para comparação robusta
_AGGREGATE_REGIONS_LOWER = {
    "world", "europe", "asia pacific", "north america",
    "central and south america", "latin america", "africa",
    "middle east and caspian", "middle east", "caspian",
    "southeast asia", "other asia pacific", "other europe",
    "advanced economies", "developing economies excl. china",
    "developing economies", "emerging economies",
    "european union", "eu27", "eu28", "g20", "g7", "oecd",
    "rest of the world", "other",
}


class IEALoader:
    """
    Loads, validates and exposes IEA GEVO data via the Aggregator.

    Parameters
    ----------
    raw_dir : Path, optional
        Directory containing the IEA CSV files. Defaults to data/raw/.
    """

    def __init__(self, raw_dir: Path | None = None):
        self.raw_dir = raw_dir or _RAW_DIR

    # ── Public API ────────────────────────────────────────────────────────

    def load_raw(self, year: Literal[2025, 2026]) -> pd.DataFrame:
        """
        Load a raw IEA CSV with no aggregation — full dimensional detail preserved.

        Returns the dataset with all original columns:
          region_country, category, parameter, mode, powertrain,
          year, unit, value, Aggregate group, source, data_version

        Primary key: region_country × category × year × parameter × mode × powertrain
        """
        path = self._resolve(year)
        df   = pd.read_csv(path)
        df   = self._standardise_columns(df)
        self._validate(df, year)
        logger.info(
            f"IEA {year} raw loaded | {df.shape[0]} rows | "
            f"{df['region_country'].nunique()} countries | "
            f"categories: {sorted(df['category'].unique())}"
        )
        return df

    def load_dataset(
        self,
        year: Literal[2025, 2026],
        category: str = "Historical",
        mode: str = "Cars",
        countries_only: bool = True,
    ) -> pd.DataFrame:
        """
        Load and aggregate a single IEA dataset into the ETIL panel format.

        Aggregation is handled by Aggregator, which correctly selects:
          - powertrain='EV'          for IEA 2026 (aggregate exists)
          - BEV + PHEV + FCEV        for IEA 2025 (no aggregate in EV sales)

        Parameters
        ----------
        year         : 2025 or 2026
        category     : 'Historical', 'Projection-CPS', or 'Projection-STEPS'
        mode         : 'Cars', 'Buses', 'Trucks', 'Vans', '2 and 3 wheelers'
        countries_only: Remove aggregate regions (World, Europe, etc.)

        Returns
        -------
        pd.DataFrame
            Columns: country, category, year, mode, ev_sales, ev_stock,
                     ev_sales_share, ev_stock_share, charging_points, data_source
        """
        if mode == "Total":
            return self.load_total_dataset(
                year,
                category=category,
                countries_only=countries_only,
            )

        df_raw    = self.load_raw(year)
        schema    = self._detect_schema(df_raw)
        agg       = Aggregator(df_raw, schema=schema)
        panel     = agg.build_panel(category=category, mode=mode)

        if countries_only:
            panel = self._filter_aggregates(panel)

        # Remove projection years from Historical panels
        if category == "Historical":
            panel = panel[~panel["year"].isin(_PROJECTION_YEARS)]

        panel = panel.sort_values(["country", "year"]).reset_index(drop=True)
        logger.info(
            f"Dataset {year} | category={category} mode={mode} | "
            f"{panel['country'].nunique()} countries | {panel.shape[0]} rows"
        )
        return panel

    def load_total_dataset(
        self,
        year: Literal[2025, 2026],
        category: str = "Historical",
        countries_only: bool = True,
    ) -> pd.DataFrame:
        """Load an all-vehicle-mode panel with denominator-weighted shares."""
        df_raw = self.load_raw(year)
        schema = self._detect_schema(df_raw)
        agg = Aggregator(df_raw, schema=schema)
        sales = agg.total_ev_all_modes(category=category)
        stock = agg.total_ev_stock_all_modes(category=category)
        charging = agg.charging_points(category=category)
        sales_share = agg.total_ev_sales_share(category=category)
        stock_share = agg.total_ev_stock_share(category=category)
        panel = (
            sales.merge(
                stock[["country", "year", "ev_stock_total"]],
                on=["country", "year"],
                how="left",
            )
            .merge(
                charging[["country", "year", "charging_points"]],
                on=["country", "year"],
                how="left",
            )
            .merge(
                sales_share[["country", "year", "ev_sales_share", "share_coverage"]],
                on=["country", "year"],
                how="left",
            )
            .merge(
                stock_share[["country", "year", "ev_stock_share", "share_coverage"]],
                on=["country", "year"],
                how="left",
                suffixes=("_sales", "_stock"),
            )
            .rename(columns={"ev_sales_total": "ev_sales", "ev_stock_total": "ev_stock"})
        )
        if countries_only:
            panel = self._filter_aggregates(panel)
        if category == "Historical":
            panel = panel[~panel["year"].isin(_PROJECTION_YEARS)]
        panel["mode"] = "Total"
        panel["share_coverage"] = panel[["share_coverage_sales", "share_coverage_stock"]].min(axis=1)
        panel["data_source"] = schema
        return panel.sort_values(["country", "year"]).reset_index(drop=True)

    def load_all_datasets(
        self,
        category: str = "Historical",
        mode: str = "Cars",
        countries_only: bool = True,
    ) -> pd.DataFrame:
        """
        Load IEA 2025 and 2026, combine with deduplication.

        Strategy: 2026 takes priority over 2025 for overlapping (country, year) pairs.
        2025-only observations (e.g. countries absent from 2026) are preserved.

        Returns
        -------
        pd.DataFrame
            Same schema as load_dataset(), with data_source='combined'.
        """
        panels = []
        for year in [2025, 2026]:
            try:
                p = self.load_dataset(year, category=category,
                                       mode=mode, countries_only=countries_only)
                panels.append(p)
            except FileNotFoundError:
                logger.warning(f"IEA {year} file not found — skipped.")

        if not panels:
            raise RuntimeError("No datasets available in data/raw/.")

        combined = (
            pd.concat(panels, ignore_index=True)
            .sort_values("data_source", ascending=False)   # 2026 first
            .drop_duplicates(subset=["country", "year"], keep="first")
            .sort_values(["country", "year"])
            .reset_index(drop=True)
        )
        combined["data_source"] = "combined"
        logger.info(
            f"Combined dataset | {combined['country'].nunique()} countries | "
            f"years {combined['year'].min()}–{combined['year'].max()} | "
            f"{combined.shape[0]} rows"
        )
        return combined

    def get_aggregator(
        self,
        year: Literal[2025, 2026],
        countries_only: bool = True,
    ) -> Aggregator:
        """
        Return a configured Aggregator for direct dimensional queries.

        Use this when you need non-standard aggregations:
          agg = loader.get_aggregator(2026)
          trucks = agg.ev_sales(category='Historical', mode='Trucks')
          breakdown = agg.breakdown_by_powertrain(mode='Cars')

        Parameters
        ----------
        countries_only : bool
            If True, filter out aggregate regions before passing to Aggregator.
            This prevents double-counting when summing across countries.
        """
        df_raw = self.load_raw(year)
        if countries_only:
            df_raw = self._filter_aggregates(df_raw, col="region_country")
        schema = self._detect_schema(df_raw)
        return Aggregator(df_raw, schema=schema)

    def available_modes(self, year: Literal[2025, 2026]) -> list:
        """Vehicle modes available in a dataset for EV sales."""
        df = self.load_raw(year)
        ev_sales = df[df["parameter"] == "EV sales"]
        vehicle_modes = {"Cars","Buses","Trucks","Vans","2 and 3 wheelers"}
        return sorted(set(ev_sales["mode"].unique()) & vehicle_modes)

    def available_projections(self, year: Literal[2025, 2026]) -> list:
        """Projection scenarios available in a dataset."""
        df = self.load_raw(year)
        return sorted(c for c in df["category"].unique() if c.startswith("Projection"))

    # ── Internal ──────────────────────────────────────────────────────────

    def _resolve(self, year: int) -> Path:
        filename = _FILES.get(year)
        if not filename:
            raise ValueError(f"Year {year} not recognised. Use 2025 or 2026.")
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"IEA {year} file not found: {path}\n"
                f"Place '{filename}' in {self.raw_dir}/"
            )
        return path

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names and enforce types."""
        df = df.copy()
        df.columns = df.columns.str.strip()
        # Ensure year is integer
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        # Ensure value is float
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    def _validate(self, df: pd.DataFrame, year: int) -> None:
        """
        Validate dimensional integrity of the raw dataset.
        Logs warnings — does not raise — to allow partial data to pass.
        """
        required_cols = {
            "region_country", "category", "parameter",
            "mode", "powertrain", "year", "value"
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"IEA {year}: missing columns: {missing}")

        # Check for unexpected categories
        unknown_cats = set(df["category"].unique()) - _VALID_CATEGORIES
        if unknown_cats:
            logger.warning(f"IEA {year}: unknown categories: {unknown_cats}")

        # Duplicate primary key check
        pk = ["region_country", "category", "year", "parameter", "mode", "powertrain"]
        dupes = df.duplicated(subset=pk, keep=False).sum()
        if dupes > 0:
            logger.warning(
                f"IEA {year}: {dupes} duplicate primary key rows detected. "
                "These will be summed during aggregation."
            )

        # Null check
        null_values = df["value"].isna().sum()
        null_pct    = null_values / len(df) * 100
        if null_pct > 20:
            logger.warning(
                f"IEA {year}: {null_pct:.1f}% null values in 'value' column."
            )

        logger.info(
            f"IEA {year} validation passed | "
            f"categories: {sorted(df['category'].unique())} | "
            f"powertrains: {sorted(df['powertrain'].unique())} | "
            f"null values: {null_pct:.1f}%"
        )

    @staticmethod
    def _detect_schema(df: pd.DataFrame) -> str:
        """
        Detect IEA schema version.

        2026: powertrain='EV' present in EV sales records
        2025: only BEV/PHEV/FCEV in EV sales; 'EV' appears only in other parameters
        """
        sales_pts = set(
            df.loc[df["parameter"] == "EV sales", "powertrain"].unique()
        )
        schema = "2026" if "EV" in sales_pts else "2025"
        logger.info(f"Schema detected: IEA {schema} (EV sales powertrains: {sales_pts})")
        return schema

    @staticmethod
    def _filter_aggregates(df: pd.DataFrame, col: str = "country") -> pd.DataFrame:
        """Remove aggregate regions using robust lowercase comparison."""
        if col not in df.columns:
            return df
        mask = df[col].str.strip().str.lower().isin(_AGGREGATE_REGIONS_LOWER)
        n_agg = mask.sum()
        if n_agg > 0:
            removed = df[mask][col].unique().tolist()
            logger.warning(f"Agregados removidos: {removed}")
        return df[~mask].reset_index(drop=True)


# ── Module-level convenience functions ───────────────────────────────────────

def load_dataset(
    year: Literal[2025, 2026],
    category: str = "Historical",
    mode: str = "Cars",
    countries_only: bool = True,
) -> pd.DataFrame:
    """Module-level shortcut for IEALoader().load_dataset()."""
    return IEALoader().load_dataset(year, category=category,
                                     mode=mode, countries_only=countries_only)


def load_all_datasets(
    category: str = "Historical",
    mode: str = "Cars",
    countries_only: bool = True,
) -> pd.DataFrame:
    """Module-level shortcut for IEALoader().load_all_datasets()."""
    return IEALoader().load_all_datasets(category=category, mode=mode,
                                          countries_only=countries_only)


def load_raw(year: Literal[2025, 2026]) -> pd.DataFrame:
    """Module-level shortcut for IEALoader().load_raw()."""
    return IEALoader().load_raw(year)
