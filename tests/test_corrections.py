import numpy as np
import pandas as pd
import pytest

from data.world_bank_client import get_fundamentals_score
from ingestion.aggregator import Aggregator
from models.scoring.factor_engine import FactorEngine


def _total_share_fixture(include_car_sales_share: bool = True) -> pd.DataFrame:
    rows = [
        # Cars: 10 EV sales at 10% => 100 total sales; 20 EV stock at 20% => 100 stock.
        dict(region_country="Testland", category="Historical", parameter="EV sales", mode="Cars", powertrain="EV", year=2024, value=10, unit="Vehicles"),
        dict(region_country="Testland", category="Historical", parameter="EV stock", mode="Cars", powertrain="EV", year=2024, value=20, unit="Vehicles"),
        # Buses: 10 EV sales at 50% => 20 total sales; 10 EV stock at 50% => 20 stock.
        dict(region_country="Testland", category="Historical", parameter="EV sales", mode="Buses", powertrain="EV", year=2024, value=10, unit="Vehicles"),
        dict(region_country="Testland", category="Historical", parameter="EV stock", mode="Buses", powertrain="EV", year=2024, value=10, unit="Vehicles"),
        dict(region_country="Testland", category="Historical", parameter="EV stock share", mode="Cars", powertrain="EV", year=2024, value=20, unit="percent"),
        dict(region_country="Testland", category="Historical", parameter="EV stock share", mode="Buses", powertrain="EV", year=2024, value=50, unit="percent"),
        dict(region_country="Testland", category="Historical", parameter="EV sales share", mode="Buses", powertrain="EV", year=2024, value=50, unit="percent"),
    ]
    if include_car_sales_share:
        rows.append(
            dict(region_country="Testland", category="Historical", parameter="EV sales share", mode="Cars", powertrain="EV", year=2024, value=10, unit="percent")
        )
    return pd.DataFrame(rows)


def test_total_shares_are_denominator_weighted():
    result = Aggregator(_total_share_fixture(), schema="2026")
    sales = result.total_ev_sales_share()
    stock = result.total_ev_stock_share()

    assert sales.loc[0, "ev_sales_share"] == pytest.approx(100 * 20 / 120)
    assert stock.loc[0, "ev_stock_share"] == pytest.approx(100 * 30 / 120)
    assert sales.loc[0, "share_coverage"] == pytest.approx(1.0)
    assert stock.loc[0, "share_coverage"] == pytest.approx(1.0)


def test_total_share_exposes_partial_coverage_instead_of_inventing_a_denominator():
    result = Aggregator(_total_share_fixture(include_car_sales_share=False), schema="2026")
    sales = result.total_ev_sales_share()

    assert pd.isna(sales.loc[0, "ev_sales_share"])
    assert sales.loc[0, "share_coverage"] == pytest.approx(0.5)


def _factor_panel() -> pd.DataFrame:
    rows = []
    for country_id in range(12):
        country = f"Country {country_id}"
        for year in range(2010, 2027):
            stock_share = 5.0 + country_id * 0.2 + (year - 2010) * 0.05
            if year == 2025 and country_id == 0:
                stock_share = 10_000.0
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "ev_sales": 100 + country_id + year - 2010,
                    "ev_stock_share": stock_share,
                    "ev_sales_share": 3.0 + country_id * 0.1,
                    "growth_rate": 2.0 + country_id * 0.05,
                }
            )
    return pd.DataFrame(rows)


def test_factor_engine_clips_future_lag_using_training_cutoff_only():
    engine = FactorEngine(weights={"penetration_stock": 1.0})
    engine.fit(_factor_panel(), calibration_end_year=2024)

    assert engine.calibration_end_year_ == 2024
    panel = engine.panel_
    future_lag = panel.loc[panel["year"] == 2026, "ev_stock_share_lag1"]
    training_lag = panel.loc[panel["year"] <= 2024, "ev_stock_share_lag1"]
    assert future_lag.max() <= training_lag.max()
    assert engine.get_ranking().shape[0] == 12


def test_factor_engine_rejects_cutoff_at_latest_year():
    with pytest.raises(ValueError, match="before the latest scoring year"):
        FactorEngine(weights={"penetration_stock": 1.0}).fit(
            _factor_panel(), calibration_end_year=2026
        )


def test_world_bank_score_normalizes_available_weights():
    score = get_fundamentals_score(
        {
            "coverage": 2,
            "gdp_per_capita": 40_000.0,
            "co2_per_capita": 0.0,
        }
    )
    gdp_norm = np.clip((40_000 - 500) / (80_000 - 500), 0, 1)
    expected = round(gdp_norm / 1.5 * 100, 1)
    assert score == expected
