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
