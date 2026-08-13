"""ETIL — institutional research terminal entry point."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import streamlit as st

from ingestion.dataset_engine import DatasetEngine
from preprocessing.cleaning import DataCleaner
from preprocessing.features import FeatureEngineer
from preprocessing.transformation import DataTransformer
from utils.logger import setup_logger

logger = setup_logger(__name__)

st.set_page_config(
    page_title="ETIL | EV Transition Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=600, show_spinner="Loading IEA historical panel…")
def get_data(dataset_key: str, vehicle_mode: str) -> pd.DataFrame:
    """Load the canonical IEA panel and run its audited processing pipeline."""
    engine = DatasetEngine()

    def load_source(key: str) -> pd.DataFrame:
        loaders = {
            "2025": lambda: engine.load_dataset(2025, vehicle_mode=vehicle_mode),
            "2026": lambda: engine.load_dataset(2026, vehicle_mode=vehicle_mode),
            "combined": lambda: engine.load_all_datasets(vehicle_mode=vehicle_mode),
        }
        if key not in loaders:
            raise ValueError(f"Unknown dataset key: {key}")
        return loaders[key]()

    actual_key = dataset_key
    try:
        raw = load_source(dataset_key)
    except Exception as error:
        logger.warning("Dataset %s failed: %s", dataset_key, error)
        raw = pd.DataFrame()
        for fallback in ("2026", "2025"):
            if fallback == dataset_key:
                continue
            try:
                raw = load_source(fallback)
                actual_key = fallback
                break
            except Exception as fallback_error:
                logger.warning("Fallback dataset %s failed: %s", fallback, fallback_error)

    if raw.empty:
        return pd.DataFrame()

    try:
        panel = raw.copy()
        panel["ev_sales"] = panel["ev_sales"].replace(0, np.nan)
        panel = DataCleaner(panel).clean()
        panel = DataTransformer(panel).add_log_transform(["ev_sales"])
        panel = FeatureEngineer(panel).engineer()
        if panel.duplicated(subset=["country", "year"]).any():
            raise ValueError("Pipeline produced duplicate country-year keys.")
    except Exception as error:
        logger.exception("Canonical data pipeline failed: %s", error)
        return pd.DataFrame()

    panel.attrs.update(
        dataset=actual_key,
        vehicle_mode=vehicle_mode,
        fallback_used=actual_key != dataset_key,
    )
    return panel


def _apply_terminal_style() -> None:
    """Apply small, shared style adjustments without changing page-level analytics."""
    st.markdown(
        """
        <style>
        :root { --etil-navy: #0F3B5E; --etil-slate: #556575; --etil-border: #D9E2EC; }
        .block-container { max-width: 1580px; padding-top: 1.6rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { border-right: 1px solid var(--etil-border); }
        [data-testid="stSidebar"] .stRadio label { font-size: 0.86rem; padding: 0.14rem 0; }
        [data-testid="stMetric"] { border-top: 2px solid var(--etil-navy); padding-top: 0.45rem; }
        [data-testid="stMetricLabel"] { font-size: 0.72rem; letter-spacing: 0.055em; text-transform: uppercase; color: var(--etil-slate); }
        [data-testid="stMetricValue"] { font-size: 1.45rem; }
        [data-testid="stTabs"] [role="tablist"] { gap: 1.25rem; border-bottom: 1px solid var(--etil-border); }
        [data-testid="stTabs"] [role="tab"] { height: 2.4rem; font-size: 0.84rem; }
        [data-testid="stDataFrame"] { border: 1px solid var(--etil-border); border-radius: 4px; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar() -> tuple[str | None, str | None, str]:
    st.sidebar.markdown("### ETIL")
    st.sidebar.caption("EV TRANSITION INTELLIGENCE · RESEARCH TERMINAL")
    st.sidebar.divider()

    pages = {
        "Market Overview": "streamlit_pages.01_Global_Monitor",
        "Country Intelligence": "streamlit_pages.02_Country_Lab",
        "Model Lab": "streamlit_pages.20_Model_Lab",
        "Signals & Rankings": "streamlit_pages.21_Signals_Research",
        "Infrastructure": "streamlit_pages.08_Infrastructure",
        "Live Intelligence": "streamlit_pages.07_EV_Pulse",
    }
    selection = st.sidebar.radio("Research area", list(pages), label_visibility="collapsed")
    module_name = pages[selection]

    if selection == "Live Intelligence":
        st.sidebar.info("This feed uses its own sourced-news workflow.")
        dataset_key, vehicle_mode = None, None
    else:
        st.sidebar.markdown("**Panel definition**")
        dataset_key = {
            "IEA 2026 historical": "2026",
            "IEA 2025 historical": "2025",
            "Combined releases": "combined",
        }[st.sidebar.selectbox("Dataset release", ["IEA 2026 historical", "IEA 2025 historical", "Combined releases"])]
        vehicle_mode = {
            "Passenger cars": "Cars",
            "Two- and three-wheelers": "2 and 3 wheelers",
            "All vehicle modes": "Total",
        }[st.sidebar.selectbox("Vehicle universe", ["Passenger cars", "Two- and three-wheelers", "All vehicle modes"])]

    st.sidebar.divider()
    st.sidebar.caption("DATA: IEA GLOBAL EV OUTLOOK · MACRO CONTEXT: WORLD BANK")
    st.sidebar.caption("Research outputs are analytical, not investment advice.")
    return dataset_key, vehicle_mode, module_name


def main() -> None:
    _apply_terminal_style()
    dataset_key, vehicle_mode, module_name = _sidebar()
    panel = pd.DataFrame()

    if dataset_key and vehicle_mode:
        panel = get_data(dataset_key, vehicle_mode)
        if panel.empty:
            st.error("The canonical IEA panel could not be loaded. Confirm the raw CSV files are available under `data/raw/`.")
            return
        if panel.attrs.get("fallback_used"):
            st.warning(f"Requested release unavailable; displaying the {panel.attrs['dataset']} fallback.")

    try:
        page = importlib.import_module(module_name)
        page.render(panel)
    except Exception as error:
        logger.exception("Page %s failed: %s", module_name, error)
        st.error("This research view could not be rendered. The error has been logged for review.")


if __name__ == "__main__":
    main()
