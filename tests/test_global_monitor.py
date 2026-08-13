import importlib

import pandas as pd


monitor = importlib.import_module("streamlit_pages.01_Global_Monitor")


def test_global_sales_timeseries_without_population_supports_volume_only():
    df = pd.DataFrame(
        {
            "year": [2023, 2023, 2024, 2024],
            "ev_sales": [100.0, 200.0, 150.0, 250.0],
        }
    )

    ts, has_population = monitor._global_sales_timeseries(df)

    assert has_population is False
    assert ts["ev_sales"].tolist() == [300.0, 400.0]
    assert ts["per_capita"].isna().all()


def test_global_sales_timeseries_computes_per_capita_when_population_exists():
    df = pd.DataFrame(
        {
            "year": [2023, 2023],
            "ev_sales": [1000.0, 3000.0],
            "population": [1_000_000.0, 3_000_000.0],
        }
    )

    ts, has_population = monitor._global_sales_timeseries(df)

    assert has_population is True
    assert ts.loc[0, "per_capita"] == 1.0
