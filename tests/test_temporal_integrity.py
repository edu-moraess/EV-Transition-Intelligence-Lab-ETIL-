import numpy as np
import pandas as pd

from models.scoring.factor_engine import FactorEngine


def _temporal_factor_panel(n_countries=12, start=2010, end=2026):
    rows = []
    for country_id in range(n_countries):
        country = f"Country {country_id}"
        for year in range(start, end + 1):
            rows.append({
                "country": country,
                "year": year,
                "ev_sales": 100 + country_id * 10 + (year - start) * (country_id + 1),
                "ev_stock_share": 2 + country_id * 0.8 + (year - start) * 0.15,
                "ev_sales_share": 1 + country_id * 0.4 + (year - start) * 0.08,
                "growth_rate": 1.5 + country_id * 0.25 + (year - start) * 0.03,
            })
    return pd.DataFrame(rows)


def test_factor_engine_target_for_cutoff_row_is_observed_by_latest_year():
    df = _temporal_factor_panel()
    engine = FactorEngine(weights={"penetration_stock": 1.0})
    engine.fit(df, calibration_end_year=2024)

    row = engine.panel_.loc[
        engine.panel_["country"].eq("Country 0") & engine.panel_["year"].eq(2024)
    ].iloc[0]
    observed_growth_2025 = df.loc[
        df["country"].eq("Country 0") & df["year"].eq(2025), "growth_rate"
    ].iloc[0]

    assert engine.as_of_year_ == 2026
    assert row["growth_t1"] == observed_growth_2025
    assert engine.calibration_end_year_ == 2024


def test_factor_engine_growth_beyond_calibration_window_does_not_change_weights():
    df = _temporal_factor_panel()
    baseline = FactorEngine()
    baseline.fit(df, calibration_end_year=2024)

    shocked = df.copy()
    shocked.loc[shocked["year"].eq(2026), "growth_rate"] *= 10000
    changed = FactorEngine()
    changed.fit(shocked, calibration_end_year=2024)

    assert baseline._calibrated_weights == changed._calibrated_weights
    pd.testing.assert_frame_equal(
        baseline.ic_by_factor_.reset_index(drop=True),
        changed.ic_by_factor_.reset_index(drop=True),
    )


def test_factor_engine_walk_forward_predictions_are_isolated_from_future_rows():
    df = _temporal_factor_panel(end=2027)
    baseline = FactorEngine().walk_forward(df, min_train_years=6)
    shocked = df.copy()
    shocked.loc[shocked["year"].eq(2026), "growth_rate"] *= 10000
    changed = FactorEngine().walk_forward(shocked, min_train_years=6)

    left = baseline.loc[baseline["origin_year"].eq(2025)].sort_values("country")
    right = changed.loc[changed["origin_year"].eq(2025)].sort_values("country")

    assert not left.empty
    assert left["country"].tolist() == right["country"].tolist()
    np.testing.assert_allclose(
        left["composite_score"].to_numpy(),
        right["composite_score"].to_numpy(),
        rtol=0,
        atol=1e-12,
    )
    assert not np.allclose(
        left["target_growth_t1"].to_numpy(),
        right["target_growth_t1"].to_numpy(),
    )


def test_factor_engine_default_cutoff_uses_only_labels_observed_by_latest_year():
    df = _temporal_factor_panel()
    engine = FactorEngine(weights={"penetration_stock": 1.0})
    engine.fit(df)

    assert engine.as_of_year_ == 2026
    assert engine.calibration_end_year_ == 2025
    row = engine.panel_.loc[
        engine.panel_["country"].eq("Country 0") & engine.panel_["year"].eq(2025)
    ].iloc[0]
    observed_growth_2026 = df.loc[
        df["country"].eq("Country 0") & df["year"].eq(2026), "growth_rate"
    ].iloc[0]
    assert row["growth_t1"] == observed_growth_2026


def test_factor_engine_lag_and_target_require_adjacent_calendar_years():
    rows = []
    for year in [2020, 2021, 2023, 2024]:
        rows.append({
            "country": "Gapland",
            "year": year,
            "ev_sales": 100 + year,
            "ev_stock_share": float(year - 2010),
            "ev_sales_share": float(year - 2015),
            "growth_rate": float(year - 2019),
        })
    panel = FactorEngine()._build_panel(
        pd.DataFrame(rows), fuel_df=None, infra_df=None, calibration_end_year=2023
    )

    row_2023 = panel.loc[panel["year"].eq(2023)].iloc[0]
    row_2021 = panel.loc[panel["year"].eq(2021)].iloc[0]
    assert pd.isna(row_2023["ev_stock_share_lag1"])
    assert pd.isna(row_2021["growth_t1"])


def test_feature_engineer_growth_and_momentum_respect_calendar_gaps():
    from preprocessing.features import FeatureEngineer

    rows = [
        {"country": "Gapland", "year": year, "ev_sales": sales}
        for year, sales in [
            (2010, 100), (2011, 110), (2012, 121),
            (2013, 133.1), (2015, 160), (2016, 176), (2017, 193.6),
        ]
    ]
    engineered = FeatureEngineer(pd.DataFrame(rows)).engineer()

    growth_2015 = engineered.loc[engineered["year"].eq(2015), "growth_rate"].iloc[0]
    growth_2016 = engineered.loc[engineered["year"].eq(2016), "growth_rate"].iloc[0]
    momentum_2017 = engineered.loc[engineered["year"].eq(2017), "momentum_3y"].iloc[0]

    assert pd.isna(growth_2015)
    assert growth_2016 == pytest.approx(10.0)
    assert pd.isna(momentum_2017)
