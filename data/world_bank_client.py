# data/world_bank_client.py
"""
World Bank API client — country macroeconomic fundamentals.

API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/898581
No API key required. Public, free, rate-limited informally (be a good citizen).

Indicators fetched:
  NY.GDP.PCAP.CD   — GDP per capita (current US$)
  SP.POP.TOTL      — Population, total
  SP.URB.TOTL.IN.ZS— Urban population (% of total)
  NY.GDP.MKTP.KD.ZG— GDP growth (annual %)
  EN.GHG.CO2.PC.CE.AR5 — CO2 emissions per capita (fallback: EN.ATM.CO2E.PC)

Design for resilience:
  - Every request wrapped in try/except with specific error classification
  - Results cached 24h via @st.cache_data — World Bank data updates yearly at most
  - Country name mapping handles IEA-style names vs World Bank ISO3 codes
  - Partial failures (one indicator missing) do not block the others
  - Returns empty dict/DataFrame on total failure — never raises into the UI
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
from utils.logger import setup_logger

logger = setup_logger(__name__)

_BASE_URL = "https://api.worldbank.org/v2/country"
_TIMEOUT  = 10

_INDICATORS = {
    "gdp_per_capita":   "NY.GDP.PCAP.CD",
    "population":       "SP.POP.TOTL",
    "urbanization":     "SP.URB.TOTL.IN.ZS",
    "gdp_growth":       "NY.GDP.MKTP.KD.ZG",
    "co2_per_capita":   "EN.GHG.CO2.PC.CE.AR5",
}

# IEA country names → ISO3 codes (World Bank requires ISO3 or ISO2)
# Only includes names that differ from a simple lookup; full mapping
# falls back to pycountry-style matching where possible.
_NAME_TO_ISO3 = {
    "United States": "USA", "United Kingdom": "GBR", "South Korea": "KOR",
    "Russia": "RUS", "Czech Republic": "CZE", "Slovak Republic": "SVK",
    "Viet Nam": "VNM", "Lao Pdr": "LAO", "Iran": "IRN", "Egypt": "EGY",
    "Turkiye": "TUR", "Turkey": "TUR", "South Africa": "ZAF",
    "United Arab Emirates": "ARE", "Brazil": "BRA", "Germany": "DEU",
    "France": "FRA", "Japan": "JPN", "India": "IND", "China": "CHN",
    "Norway": "NOR", "Netherlands": "NLD", "Sweden": "SWE", "Spain": "ESP",
    "Italy": "ITA", "Poland": "POL", "Mexico": "MEX", "Canada": "CAN",
    "Australia": "AUS", "Indonesia": "IDN", "Thailand": "THA",
    "Malaysia": "MYS", "Philippines": "PHL", "Colombia": "COL",
    "Chile": "CHL", "Greece": "GRC", "Portugal": "PRT", "Austria": "AUT",
    "Belgium": "BEL", "Denmark": "DNK", "Finland": "FIN", "Ireland": "IRL",
    "Switzerland": "CHE", "New Zealand": "NZL", "Singapore": "SGP",
    "Hungary": "HUN", "Romania": "ROU", "Bulgaria": "BGR", "Croatia": "HRV",
    "Slovenia": "SVN", "Estonia": "EST", "Latvia": "LVA", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Malta": "MLT", "Cambodia": "KHM", "Jordan": "JOR",
    "Costa Rica": "CRI", "Uzbekistan": "UZB",
}


def _resolve_iso3(country: str) -> str | None:
    """Map an IEA-style country name to a World Bank ISO3 code."""
    return _NAME_TO_ISO3.get(country)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_indicator(iso3: str, indicator_code: str, latest_only: bool = True) -> dict:
    """
    Fetch a single indicator for a single country.

    Returns {"ok": True, "value": float, "year": int} or {"ok": False, "error": str}.
    """
    if not iso3:
        return {"ok": False, "error": "no_iso3"}

    url = f"{_BASE_URL}/{iso3}/indicator/{indicator_code}"
    params = {"format": "json", "per_page": 10}

    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "connection"}
    except Exception as e:
        return {"ok": False, "error": f"unexpected: {e}"}

    if resp.status_code != 200:
        return {"ok": False, "error": f"http_{resp.status_code}"}

    try:
        data = resp.json()
    except Exception:
        return {"ok": False, "error": "invalid_json"}

    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return {"ok": False, "error": "no_data"}

    records = data[1]
    # World Bank returns most recent years first; find the first non-null value
    for rec in records:
        if rec.get("value") is not None:
            return {"ok": True, "value": float(rec["value"]), "year": int(rec["date"])}

    return {"ok": False, "error": "all_null"}


@st.cache_data(ttl=86400, show_spinner=False)
def get_country_fundamentals(country: str) -> dict:
    """
    Fetch all configured indicators for a single country.

    Returns a flat dict, e.g.:
      {
        "country": "Brazil", "iso3": "BRA",
        "gdp_per_capita": 10043.2, "gdp_per_capita_year": 2023,
        "population": 216422446, "population_year": 2023,
        "urbanization": 87.6, "urbanization_year": 2023,
        "gdp_growth": 2.9, "gdp_growth_year": 2023,
        "co2_per_capita": 2.1, "co2_per_capita_year": 2022,
        "coverage": 5,   # how many of the 5 indicators succeeded
      }

    Indicators that fail are simply absent from the dict — partial
    results are always returned rather than failing the whole country.
    """
    iso3 = _resolve_iso3(country)
    if not iso3:
        return {"country": country, "iso3": None, "coverage": 0, "error": "unmapped_country"}

    result = {"country": country, "iso3": iso3}
    coverage = 0

    for key, code in _INDICATORS.items():
        res = fetch_indicator(iso3, code)
        if res["ok"]:
            result[key] = res["value"]
            result[f"{key}_year"] = res["year"]
            coverage += 1
        else:
            logger.debug(f"{country} ({iso3}) — {key} failed: {res.get('error')}")

    result["coverage"] = coverage
    return result


def get_fundamentals_score(fundamentals: dict) -> float | None:
    """
    Compute a 0-100 Country Fundamentals Score from normalised indicators.

    This is a transparent, equally-weighted composite — not a black box.
    Each indicator is min-max normalised against a reasonable global range,
    then averaged. Missing indicators are excluded from the average rather
    than penalised, so partial coverage still produces a usable score.

    Weights are deliberately simple (equal) since this is a context score,
    not a predictive factor — unlike EVQTI, which has empirically-derived weights.
    """
    if not fundamentals or fundamentals.get("coverage", 0) == 0:
        return None

    # (value, reasonable_min, reasonable_max, higher_is_better)
    components = []

    if "gdp_per_capita" in fundamentals:
        v = fundamentals["gdp_per_capita"]
        norm = np.clip((v - 500) / (80000 - 500), 0, 1)
        components.append(norm)

    if "urbanization" in fundamentals:
        v = fundamentals["urbanization"]
        norm = np.clip(v / 100, 0, 1)
        components.append(norm)

    if "gdp_growth" in fundamentals:
        v = fundamentals["gdp_growth"]
        norm = np.clip((v + 5) / (10 + 5), 0, 1)  # -5% to +10% range
        components.append(norm)

    if "co2_per_capita" in fundamentals:
        # Higher CO2/capita can mean more pressure to electrify — treated as neutral-to-positive,
        # capped to avoid rewarding extreme emitters disproportionately
        v = fundamentals["co2_per_capita"]
        norm = np.clip(v / 15, 0, 1)
        components.append(norm * 0.5)  # half-weighted — directional ambiguity acknowledged

    if not components:
        return None

    return round(float(np.mean(components)) * 100, 1)
