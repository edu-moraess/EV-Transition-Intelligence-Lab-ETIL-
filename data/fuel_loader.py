# data/fuel_loader.py
"""
Fuel Price Loader — IEA Energy Prices Monthly Excerpt.

Source  : IEA Energy Prices (Jun 2026)
Coverage: 33 countries · Jan 2015 – May 2026 · USD/litre
Products: Gasoline · Diesel

Ukraine is excluded: only 135 months of data, irregular coverage.

Derived metrics (all in USD/litre unless noted):
  gasoline_latest   — most recent available month
  diesel_latest     — most recent available month
  gasoline_avg_12m  — 12-month rolling average
  gasoline_yoy      — year-on-year change (%) vs same month last year
  fuel_volatility   — std dev of gasoline price over last 12 months
  diesel_gas_spread — diesel minus gasoline (latest)

Annual aggregation (for cross-country EV vs fuel comparison):
  gasoline_annual   — calendar-year mean (USD/litre)
  diesel_annual     — calendar-year mean (USD/litre)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

_FUEL_PATH   = Path(__file__).parent / "raw" / "fuel_prices_iea.csv"
_EXCLUDE     = {"Ukraine"}   # insufficient data coverage


def load_fuel_prices() -> pd.DataFrame:
    """
    Load and clean the IEA fuel price CSV.

    Returns
    -------
    pd.DataFrame
        Columns: date, country, diesel_usd, gasoline_usd
        Sorted by (country, date). Ukraine excluded.
    """
    if not _FUEL_PATH.exists():
        logger.warning(f"Fuel price file not found: {_FUEL_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(_FUEL_PATH)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df["country"].isin(_EXCLUDE)].copy()
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    logger.info(
        f"Fuel prices loaded: {df['country'].nunique()} countries · "
        f"{df['date'].min().strftime('%b %Y')} → {df['date'].max().strftime('%b %Y')}"
    )
    return df


def get_fuel_summary(country: str) -> dict:
    """
    Return a summary dict of key fuel metrics for a given country.
    Returns empty dict if country not covered.
    """
    df = load_fuel_prices()
    if df.empty:
        return {}

    c = df[df["country"] == country].copy()
    if c.empty:
        return {}

    c = c.sort_values("date")

    # Latest non-null values
    gas_series = c["gasoline_usd"].dropna()
    die_series = c["diesel_usd"].dropna()

    gas_latest = float(gas_series.iloc[-1]) if not gas_series.empty else np.nan
    die_latest = float(die_series.iloc[-1]) if not die_series.empty else np.nan
    latest_date = c["date"].iloc[-1].strftime("%b %Y")

    # 12-month rolling average of gasoline
    gas_12m = gas_series.iloc[-12:].mean() if len(gas_series) >= 12 else gas_series.mean()

    # YoY change: compare latest with same month 1 year ago
    if len(gas_series) >= 13:
        gas_prev_yr = float(gas_series.iloc[-13])
        gas_yoy = (gas_latest - gas_prev_yr) / gas_prev_yr * 100
    else:
        gas_yoy = np.nan

    # 12-month volatility (annualised std dev of monthly returns)
    if len(gas_series) >= 13:
        returns = gas_series.pct_change().dropna().iloc[-12:]
        volatility = float(returns.std() * np.sqrt(12) * 100)
    else:
        volatility = np.nan

    # Diesel-gasoline spread (latest)
    spread = die_latest - gas_latest if not np.isnan(die_latest) else np.nan

    return {
        "country":          country,
        "latest_date":      latest_date,
        "gasoline_latest":  round(gas_latest, 3),
        "diesel_latest":    round(die_latest, 3),
        "gasoline_avg_12m": round(gas_12m, 3),
        "gasoline_yoy_pct": round(gas_yoy, 1) if not np.isnan(gas_yoy) else None,
        "fuel_volatility":  round(volatility, 1) if not np.isnan(volatility) else None,
        "diesel_gas_spread": round(spread, 3) if not np.isnan(spread) else None,
    }


def get_annual_fuel(countries: list | None = None) -> pd.DataFrame:
    """
    Return annual average gasoline and diesel prices (USD/litre).
    Useful for year-level cross-country comparison with EV data.
    """
    df = load_fuel_prices()
    if df.empty:
        return pd.DataFrame()

    df["year"] = df["date"].dt.year
    if countries:
        df = df[df["country"].isin(countries)]

    annual = (
        df.groupby(["country", "year"])
        .agg(gasoline_avg=("gasoline_usd", "mean"),
             diesel_avg=("diesel_usd", "mean"))
        .reset_index()
        .round(3)
    )
    return annual


def get_monthly_fuel(country: str) -> pd.DataFrame:
    """Return full monthly time series for a given country."""
    df = load_fuel_prices()
    if df.empty:
        return pd.DataFrame()
    return df[df["country"] == country].copy()


def covered_countries() -> list:
    """Return sorted list of countries with fuel price data."""
    df = load_fuel_prices()
    return sorted(df["country"].unique().tolist()) if not df.empty else []
