import numpy as np
import pandas as pd
import pytest

from ingestion.aggregator import Aggregator
from models.diffusion.bass import BassDiffusion
from models.forecasting.monte_carlo import MonteCarloSimulator
from models.forecasting.regime_forecast import RegimeAwareForecastEngine
from preprocessing.cleaning import DataCleaner


def test_2025_charging_points_accepts_ev_mode():
    raw = pd.DataFrame(
        {
            "region_country": ["A"],
            "category": ["Historical"],
            "year": [2024],
            "parameter": ["EV charging points"],
            "mode": ["EV"],
            "powertrain": ["Publicly available slow"],
            "value": [125.0],
        }
    )
    result = Aggregator(raw, schema="2025").charging_points()
    assert result.loc[0, "charging_points"] == 125.0


def test_2026_component_fallback_only_fills_missing_aggregate_keys():
    raw = pd.DataFrame(
        {
            "region_country": ["A", "A", "B", "B"],
            "category": ["Historical"] * 4,
            "year": [2024] * 4,
            "parameter": ["EV sales"] * 4,
            "mode": ["Cars"] * 4,
            "powertrain": ["EV", "BEV", "BEV", "PHEV"],
            "value": [100.0, 90.0, 10.0, 5.0],
        }
    )
    result = Aggregator(raw, schema="2026").ev_sales()
    totals = result.set_index("country")["ev_sales"].to_dict()
    assert totals == {"A": 100.0, "B": 15.0}


def test_cleaner_does_not_impute_missing_sales():
    panel = pd.DataFrame(
        {
            "country": ["A", "A"],
            "year": [2020, 2021],
            "ev_sales": [10.0, np.nan],
        }
    )
    cleaned = DataCleaner(panel).clean()
    assert pd.isna(cleaned.loc[cleaned["year"] == 2021, "ev_sales"].iloc[0])


def test_bass_enforces_monotone_cumulative_input_and_market_bound():
    model = BassDiffusion()
    with pytest.raises(ValueError, match="non-decreasing"):
        model.fit(np.arange(1, 5), np.array([1.0, 5.0, 4.0, 8.0]))
    observed = np.array([0.0, 10.0, 35.0, 70.0, 120.0])
    M, _, _ = model.fit(np.arange(1, 6), observed)
    assert M >= observed.max()
    assert np.all(np.isfinite(model.predict(np.arange(1, 15))))


def test_monte_carlo_paths_are_reproducible_with_a_local_seed():
    class FakeGP:
        def sample_paths(self, X, n_samples, seed=None):
            return np.random.default_rng(seed).normal(size=(n_samples, len(X)))

    inputs = np.arange(5).reshape(-1, 1)
    first = MonteCarloSimulator(n_simulations=100, seed=7).simulate_from_gp(FakeGP(), inputs)
    second = MonteCarloSimulator(n_simulations=100, seed=7).simulate_from_gp(FakeGP(), inputs)
    assert np.array_equal(first, second)


def test_regime_forecast_has_positive_first_year_increment_and_ordered_quantiles():
    years = np.arange(2018, 2026)
    panel = pd.DataFrame(
        {
            "year": years,
            "ev_sales": [10, 14, 22, 37, 55, 82, 120, 175],
            "ev_stock_share": np.linspace(0.1, 2.5, len(years)),
            "growth_rate": [np.nan, 40, 57, 68, 49, 49, 46, 46],
        }
    )
    result = RegimeAwareForecastEngine(n_simulations=200, seed=3).forecast(panel, horizon=4, regime_id=1)
    forecast = result["baseline"]
    assert (forecast["median"] >= 0).all()
    assert forecast.loc[0, "median"] > 0
    assert (forecast["p10"] <= forecast["median"]).all()
    assert (forecast["median"] <= forecast["p90"]).all()
