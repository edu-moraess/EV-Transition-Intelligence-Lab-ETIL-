import numpy as np
import pandas as pd
import pytest

from preprocessing.features import FeatureEngineer


def calculate(rows):
    engineer = FeatureEngineer(pd.DataFrame(rows))
    engineer._add_volatility()
    return engineer.df


def volatility_at(result, country, year):
    row = result.loc[
        result["country"].eq(country) & result["year"].eq(year),
        "volatility",
    ]
    assert len(row) == 1
    return row.iloc[0]


def test_volatility_uses_four_consecutive_calendar_positions():
    result = calculate([
        {"country": "A", "year": 2010, "growth_rate": np.nan},
        {"country": "A", "year": 2011, "growth_rate": 1.0},
        {"country": "A", "year": 2012, "growth_rate": 3.0},
        {"country": "A", "year": 2013, "growth_rate": 5.0},
        {"country": "A", "year": 2014, "growth_rate": 7.0},
    ])

    assert volatility_at(result, "A", 2013) == pytest.approx(
        np.std([1.0, 3.0, 5.0], ddof=1)
    )
    assert volatility_at(result, "A", 2014) == pytest.approx(
        np.std([1.0, 3.0, 5.0, 7.0], ddof=1)
    )


def test_missing_calendar_year_is_not_replaced_by_previous_observation():
    result = calculate([
        {"country": "A", "year": 2010, "growth_rate": np.nan},
        {"country": "A", "year": 2011, "growth_rate": 1.0},
        {"country": "A", "year": 2012, "growth_rate": 3.0},
        {"country": "A", "year": 2015, "growth_rate": np.nan},
        {"country": "A", "year": 2016, "growth_rate": 5.0},
        {"country": "A", "year": 2017, "growth_rate": 9.0},
    ])

    assert pd.isna(volatility_at(result, "A", 2016))
    assert volatility_at(result, "A", 2017) == pytest.approx(
        np.std([5.0, 9.0], ddof=1)
    )


def test_one_valid_observation_returns_nan():
    result = calculate([
        {"country": "A", "year": 2020, "growth_rate": np.nan},
        {"country": "A", "year": 2021, "growth_rate": np.nan},
        {"country": "A", "year": 2022, "growth_rate": 2.0},
        {"country": "A", "year": 2023, "growth_rate": np.nan},
    ])

    assert pd.isna(volatility_at(result, "A", 2023))


def test_two_valid_observations_use_sample_standard_deviation():
    result = calculate([
        {"country": "A", "year": 2020, "growth_rate": np.nan},
        {"country": "A", "year": 2021, "growth_rate": 1.0},
        {"country": "A", "year": 2022, "growth_rate": np.nan},
        {"country": "A", "year": 2023, "growth_rate": 3.0},
    ])

    assert volatility_at(result, "A", 2023) == pytest.approx(
        np.std([1.0, 3.0], ddof=1)
    )


def test_missing_growth_rate_in_present_year_remains_missing():
    result = calculate([
        {"country": "A", "year": 2020, "growth_rate": 1.0},
        {"country": "A", "year": 2021, "growth_rate": np.nan},
        {"country": "A", "year": 2022, "growth_rate": 5.0},
        {"country": "A", "year": 2023, "growth_rate": 7.0},
    ])

    assert volatility_at(result, "A", 2023) == pytest.approx(
        np.std([1.0, 5.0, 7.0], ddof=1)
    )


def test_countries_have_independent_volatility_windows():
    result = calculate([
        {"country": "A", "year": 2020, "growth_rate": np.nan},
        {"country": "A", "year": 2021, "growth_rate": 1.0},
        {"country": "A", "year": 2022, "growth_rate": 3.0},
        {"country": "A", "year": 2023, "growth_rate": 5.0},
        {"country": "B", "year": 2020, "growth_rate": np.nan},
        {"country": "B", "year": 2021, "growth_rate": 10.0},
        {"country": "B", "year": 2022, "growth_rate": 10.0},
        {"country": "B", "year": 2023, "growth_rate": 10.0},
    ])

    assert volatility_at(result, "A", 2023) == pytest.approx(2.0)
    assert volatility_at(result, "B", 2023) == pytest.approx(0.0)


def test_unsorted_input_is_processed_chronologically():
    result = calculate([
        {"country": "A", "year": 2022, "growth_rate": 3.0},
        {"country": "A", "year": 2020, "growth_rate": np.nan},
        {"country": "A", "year": 2023, "growth_rate": 5.0},
        {"country": "A", "year": 2021, "growth_rate": 1.0},
    ])

    assert volatility_at(result, "A", 2023) == pytest.approx(2.0)


def test_first_years_require_minimum_observations():
    result = calculate([
        {"country": "A", "year": 2020, "growth_rate": np.nan},
        {"country": "A", "year": 2021, "growth_rate": 1.0},
        {"country": "A", "year": 2022, "growth_rate": 3.0},
    ])

    assert pd.isna(volatility_at(result, "A", 2021))
    assert volatility_at(result, "A", 2022) == pytest.approx(
        np.std([1.0, 3.0], ddof=1)
    )
