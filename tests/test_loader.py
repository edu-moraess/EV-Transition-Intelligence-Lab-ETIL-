# tests/test_loader.py
"""
Tests for ingestion/loader.py — schema detection, validation, dataset combination.

Replaces the original placeholder test (instantiation only) with regression
tests that codify the manual validations done during development.
"""

import pytest
import pandas as pd
import numpy as np
from ingestion.loader import IEALoader


def _make_2026_raw():
    return pd.DataFrame([
        dict(region_country="Norway", category="Historical", parameter="EV sales",
             mode="Cars", powertrain="EV", year=2023, value=110000, unit="Vehicles"),
        dict(region_country="Norway", category="Historical", parameter="EV sales",
             mode="Cars", powertrain="BEV", year=2023, value=95000, unit="Vehicles"),
    ])


def _make_2025_raw():
    return pd.DataFrame([
        dict(region_country="Germany", category="Historical", parameter="EV sales",
             mode="Cars", powertrain="BEV", year=2022, value=40000, unit="Vehicles"),
        dict(region_country="Germany", category="Historical", parameter="EV sales",
             mode="Cars", powertrain="PHEV", year=2022, value=15000, unit="Vehicles"),
    ])


class TestSchemaDetection:

    def test_detects_2026_schema_when_ev_aggregate_present_in_sales(self):
        df = _make_2026_raw()
        schema = IEALoader._detect_schema(df)
        assert schema == "2026"

    def test_detects_2025_schema_when_no_ev_aggregate_in_sales(self):
        df = _make_2025_raw()
        schema = IEALoader._detect_schema(df)
        assert schema == "2025"

    def test_ev_powertrain_in_other_parameters_does_not_trigger_false_2026(self):
        """Regression test for the original schema-detection bug:
        powertrain='EV' appearing only in charging-points rows must not
        cause a 2025 dataset to be misclassified as 2026."""
        df = pd.DataFrame([
            dict(region_country="Germany", category="Historical", parameter="EV sales",
                 mode="Cars", powertrain="BEV", year=2022, value=40000, unit="Vehicles"),
            dict(region_country="Germany", category="Historical", parameter="EV charging points",
                 mode="EV", powertrain="EV", year=2022, value=800, unit="Chargers"),
        ])
        schema = IEALoader._detect_schema(df)
        assert schema == "2025"  # must NOT be misdetected as 2026


class TestLoaderValidation:

    def test_validate_raises_on_missing_required_columns(self):
        loader = IEALoader()
        df = pd.DataFrame({"country": ["Norway"], "value": [100]})
        with pytest.raises(ValueError):
            loader._validate(df, 2026)

    def test_validate_passes_with_all_required_columns(self):
        loader = IEALoader()
        df = _make_2026_raw()
        # Should not raise
        loader._validate(df, 2026)


class TestLoaderInstantiation:

    def test_loader_instantiation(self):
        loader = IEALoader()
        assert loader is not None

    def test_loader_has_raw_dir_configured(self):
        loader = IEALoader()
        assert loader.raw_dir is not None


class TestTotalDatasetCoverage:
    def test_combined_coverage_is_missing_if_either_component_is_missing(monkeypatch):
        class FakeAggregator:
            def __init__(self, *_args, **_kwargs):
                pass

            def total_ev_all_modes(self, category):
                return pd.DataFrame({
                    "country": ["Norway"], "year": [2025],
                    "ev_sales_total": [100.0],
                })

            def total_ev_stock_all_modes(self, category):
                return pd.DataFrame({
                    "country": ["Norway"], "year": [2025],
                    "ev_stock_total": [500.0],
                })

            def charging_points(self, category):
                return pd.DataFrame({
                    "country": ["Norway"], "year": [2025],
                    "charging_points": [10.0],
                })

            def total_ev_sales_share(self, category):
                return pd.DataFrame({
                    "country": ["Norway"], "year": [2025],
                    "ev_sales_share": [20.0], "share_coverage": [1.0],
                })

            def total_ev_stock_share(self, category):
                return pd.DataFrame({
                    "country": ["Norway"], "year": [2025],
                    "ev_stock_share": [float("nan")], "share_coverage": [float("nan")],
                })

        monkeypatch.setattr("ingestion.loader.Aggregator", FakeAggregator)
        monkeypatch.setattr(
            IEALoader, "load_raw",
            lambda self, year: pd.DataFrame({"parameter": ["EV sales"]}),
        )

        result = IEALoader().load_total_dataset(2026)

        assert pd.isna(result.loc[0, "share_coverage"])
        assert pd.isna(result.loc[0, "ev_stock_share"])
