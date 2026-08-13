from data import world_bank_client as client


def test_country_fundamentals_include_retrieval_and_observation_metadata(monkeypatch):
    client.get_country_fundamentals.clear()
    monkeypatch.setattr(client, "_resolve_iso3", lambda country: "BRA")
    monkeypatch.setattr(
        client,
        "fetch_indicator",
        lambda iso3, code: {"ok": True, "value": 1.0, "year": 2025},
    )

    result = client.get_country_fundamentals("Brazil")

    assert result["iso3"] == "BRA"
    assert result["coverage"] == len(client._INDICATORS)
    assert result["population_year"] == 2025
    assert "retrieved_at" in result
    assert result["retrieved_at"].endswith("+00:00")
    client.get_country_fundamentals.clear()
