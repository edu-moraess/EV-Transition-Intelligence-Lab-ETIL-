import pandas as pd
import pytest

from ingestion.loader import IEALoader


def make_panel(rows):
    return pd.DataFrame(
        rows,
        columns=["country", "year", "value", "data_source", "source_release"]
    )


def test_combined_preserves_release_and_2026_precedence(monkeypatch):
    loader = IEALoader()

    panels = {
        2025: make_panel([
            ("Brazil", 2020, 100, "2025", "2025"),
            ("Brazil", 2021, 110, "2025", "2025"),
            ("Chile", 2020, 50, "2025", "2025"),
        ]),
        2026: make_panel([
            ("Brazil", 2020, 105, "2026", "2026"),
            ("Brazil", 2022, 120, "2026", "2026"),
        ]),
    }

    monkeypatch.setattr(
        loader,
        "load_dataset",
        lambda year, **kwargs: panels[year].copy()
    )

    result = loader.load_all_datasets()

    assert result["data_source"].eq("combined").all()

    brazil_2020 = result[
        (result["country"] == "Brazil") &
        (result["year"] == 2020)
    ].iloc[0]

    assert brazil_2020["value"] == 105
    assert brazil_2020["source_release"] == "2026"

    brazil_2021 = result[
        (result["country"] == "Brazil") &
        (result["year"] == 2021)
    ].iloc[0]

    assert brazil_2021["source_release"] == "2025"

    chile_2020 = result[
        (result["country"] == "Chile") &
        (result["year"] == 2020)
    ].iloc[0]

    assert chile_2020["source_release"] == "2025"


def test_combined_preserves_2025_when_2026_is_unavailable(monkeypatch):
    loader = IEALoader()

    panel_2025 = make_panel([
        ("Brazil", 2020, 100, "2025", "2025"),
    ])

    def fake_load(year, **kwargs):
        if year == 2026:
            raise FileNotFoundError("2026 unavailable")
        return panel_2025.copy()

    monkeypatch.setattr(loader, "load_dataset", fake_load)

    result = loader.load_all_datasets()

    assert len(result) == 1
    assert result.iloc[0]["source_release"] == "2025"
    assert result.iloc[0]["data_source"] == "combined"


def test_combined_raises_when_no_release_is_available(monkeypatch):
    loader = IEALoader()

    def unavailable(year, **kwargs):
        raise FileNotFoundError(f"{year} unavailable")

    monkeypatch.setattr(loader, "load_dataset", unavailable)

    with pytest.raises(RuntimeError, match="No datasets available"):
        loader.load_all_datasets()
