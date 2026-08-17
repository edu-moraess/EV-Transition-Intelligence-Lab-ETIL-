# app.py
"""
EV Transition Intelligence Lab (ETIL) — main entry point.

Theme    : Streamlit native (config.toml) — no hardcoded colours in Python
Dataset  : IEA Global EV Outlook 2025 / 2026 / Combined
Language : English throughout
"""
import streamlit as st
import pandas as pd
import numpy as np
import importlib
from ingestion.dataset_engine import DatasetEngine
from preprocessing.cleaning import DataCleaner
from preprocessing.transformation import DataTransformer
from preprocessing.features import FeatureEngineer
from utils.logger import setup_logger

logger = setup_logger(__name__)

st.set_page_config(
    page_title="EV Transition Intelligence Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw data through cleaning, transformation and feature engineering.
    Returns a sanitised DataFrame with derived analytics columns.
    """
    # ── 1. Handle zeros in ev_sales (log will be -inf) ─────────────────────────
    if "ev_sales" in df_raw.columns:
        df_raw["ev_sales"] = df_raw["ev_sales"].replace(0, np.nan)

    # ── 2. Main pipeline ──────────────────────────────────────────────────────
    df = DataCleaner(df_raw).clean()
    df = DataTransformer(df).add_log_transform(["ev_sales"])
    df = FeatureEngineer(df).engineer()

    assert df.duplicated(subset=["country", "year"]).sum() == 0, "Pipeline produced duplicate country-year entries."
    return df


@st.cache_data(ttl=600, show_spinner="Loading IEA data...")
def get_data(dataset_key: str, vehicle_mode: str) -> pd.DataFrame:
    """Load and process the EV dataset, with automatic fallback."""
    engine = DatasetEngine()

    def _try_load(key: str) -> pd.DataFrame:
        if key == "2025":
            return engine.load_dataset(2025, vehicle_mode=vehicle_mode)
        elif key == "2026":
            return engine.load_dataset(2026, vehicle_mode=vehicle_mode)
        elif key == "combined":
            return engine.load_all_datasets(vehicle_mode=vehicle_mode)
        else:
            raise ValueError(f"Unknown dataset key: '{key}'")

    df_raw = pd.DataFrame()
    actual_key = dataset_key
    try:
        df_raw = _try_load(dataset_key)
    except Exception as e:
        logger.warning(f"Primary dataset '{dataset_key}' failed: {e}. Trying fallback.")
        fallback_order = [k for k in ["2026", "2025"] if k != dataset_key]
        for fb_key in fallback_order:
            try:
                df_raw = _try_load(fb_key)
                actual_key = fb_key
                logger.info(f"Fallback to dataset '{fb_key}' succeeded.")
                break
            except Exception as fb_e:
                logger.warning(f"Fallback '{fb_key}' also failed: {fb_e}")

    if df_raw.empty:
        logger.error("All dataset loading attempts failed.")
        return pd.DataFrame()

    try:
        df = _run_pipeline(df_raw)
    except Exception as e:
        logger.exception(f"Pipeline processing failed: {e}")
        return pd.DataFrame()

    if "ev_sales" not in df.columns or df["ev_sales"].notna().sum() == 0:
        logger.error("Processed dataset has no valid ev_sales values.")
        return pd.DataFrame()

    df.attrs["dataset"]      = actual_key
    df.attrs["vehicle_mode"] = vehicle_mode
    df.attrs["fallback_used"] = actual_key != dataset_key
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> tuple[str | None, str | None, str]:
    st.sidebar.markdown(
        "<h1 style='margin:0 0 0.1rem 0; font-size:1.5rem; font-weight:700;'>"
        "ETIL</h1>"
        "<p style='margin:0 0 0.8rem 0; opacity:0.55; font-size:0.82rem;'>"
        "EV Transition Intelligence Lab</p>",
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        "<p style='font-size:0.65rem; text-transform:uppercase; "
        "letter-spacing:0.08em; opacity:0.4; margin:0 0 0.2rem 0;'>"
        "Navigation</p>",
        unsafe_allow_html=True
    )
    pages = {
        "🌍  Global Monitor":       "streamlit_pages.01_Global_Monitor",
        "🔬  Country Lab":          "streamlit_pages.02_Country_Lab",
        "🧠  Diffusion Engine":      "streamlit_pages.03_Diffusion_Engine",
        "🔄  Regime Detection":     "streamlit_pages.04_Regime_Detection",
        "📊  Forecast Lab":         "streamlit_pages.05_Forecast_Lab",
        "🏆  EVQTI Ranking":        "streamlit_pages.06_EVQTI_Ranking",
        "📰  EV Pulse":             "streamlit_pages.07_EV_Pulse",
        "🔌  Infrastructure":       "streamlit_pages.08_Infrastructure",
        "🎯  Scenarios":            "streamlit_pages.09_Scenarios",
        "🌐  ETII Composite Index": "streamlit_pages.10_ETII",
        "🔬  Research Intelligence": "streamlit_pages.11_Research_Intelligence",
    }
    selection = st.sidebar.radio(
        "Navigation",
        list(pages.keys()),
        key="nav_radio",
        label_visibility="collapsed",
    )
    module_name = pages[selection]

    is_independent = module_name.endswith(("07_EV_Pulse", "08_Infrastructure"))

    if not is_independent:
        st.sidebar.markdown(
            "<p style='font-size:0.65rem; text-transform:uppercase; "
            "letter-spacing:0.08em; opacity:0.4; margin:0.6rem 0 0.2rem 0;'>"
            "Dataset</p>",
            unsafe_allow_html=True
        )
        dataset_option = st.sidebar.selectbox(
            "Dataset",
            ["IEA 2026", "IEA 2025", "Combined (2025+2026)"],
            index=0,
            key="dataset_select",
            label_visibility="collapsed",
        )
        dataset_key = {
            "IEA 2026":              "2026",
            "IEA 2025":              "2025",
            "Combined (2025+2026)":  "combined",
        }[dataset_option]

        st.sidebar.markdown(
            "<p style='font-size:0.65rem; text-transform:uppercase; "
            "letter-spacing:0.08em; opacity:0.4; margin:0.6rem 0 0.2rem 0;'>"
            "Vehicle Mode</p>",
            unsafe_allow_html=True
        )
        vehicle_option = st.sidebar.selectbox(
            "Vehicle mode",
            ["Cars", "2 and 3 wheelers", "Total (all modes)"],
            index=0,
            key="vehicle_select",
            label_visibility="collapsed",
        )
        vehicle_mode = {
            "Cars":                "Cars",
            "2 and 3 wheelers":    "2 and 3 wheelers",
            "Total (all modes)":   "Total",
        }[vehicle_option]
    else:
        dataset_key = None
        vehicle_mode = None
        st.sidebar.info("ℹ️ This page does not require the EV dataset.")

    st.sidebar.markdown("<hr style='margin:0.8rem 0; opacity:0.15;'>", unsafe_allow_html=True)

    st.sidebar.markdown(
        "<div style='font-size:0.72rem; opacity:0.45; line-height:1.5;'>"
        "<b style='opacity:0.8;'>Carlos Eduardo</b><br>"
        "Quant Data Scientist<br>"
        "<a href='https://www.linkedin.com/in/carlos-eduardo-2753a92b5' "
        "style='color:inherit; text-decoration:underline;'>LinkedIn</a> · "
        "<a href='https://github.com/edu-moraess' "
        "style='color:inherit; text-decoration:underline;'>GitHub</a><br><br>"
        "</div>",
        unsafe_allow_html=True
    )

    return dataset_key, vehicle_mode, module_name


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dataset_key, vehicle_mode, module_name = _sidebar()

    df = pd.DataFrame()
    if dataset_key is not None and vehicle_mode is not None:
        df = get_data(dataset_key, vehicle_mode)

        if df.empty:
            st.error(
                "No EV data loaded. Place the IEA CSV files in `data/raw/` and reload.\n\n"
                "Expected files:\n"
                "- `data/raw/EV_Data_Explorer_2025_GEVO_clean.csv`\n"
                "- `data/raw/EV_Data_Explorer_2026_GEVO_clean.csv`"
            )
            return

        if df.attrs.get("fallback_used"):
            st.info(
                f"⚠️ The selected dataset could not be loaded – "
                f"showing **{df.attrs.get('dataset')}** instead."
            )

    try:
        page = importlib.import_module(module_name)
        if hasattr(page, "render"):
            page.render(df)
        else:
            st.error(f"Module `{module_name}` has no `render()` function.")
    except Exception as e:
        import traceback
        st.error(f"Page error: {e}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main() 