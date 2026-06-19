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
    df = DataCleaner(df_raw).clean()
    df = DataTransformer(df).add_log_transform(["ev_sales"])
    df = FeatureEngineer(df).engineer()

    # Use ev_stock_share as penetration where available (correct IEA metric)
    if "ev_stock_share" in df.columns:
        mask = df["ev_stock_share"].notna() & (df["ev_stock_share"] > 0)
        df.loc[mask, "penetration"] = df.loc[mask, "ev_stock_share"]

    # Propagate ev_sales_share from raw (not processed by FeatureEngineer)
    if "ev_sales_share" in df_raw.columns:
        share_raw = df_raw[["country", "year", "ev_sales_share"]].drop_duplicates()
        df = df.merge(share_raw, on=["country", "year"], how="left", suffixes=("", "_raw"))
        if "ev_sales_share_raw" in df.columns:
            df.drop(columns=["ev_sales_share_raw"], inplace=True, errors="ignore")

    return df


@st.cache_data(ttl=600, show_spinner="Loading IEA data...")
def get_data(dataset_key: str, vehicle_mode: str) -> pd.DataFrame:
    engine = DatasetEngine()
    try:
        if dataset_key == "2025":
            df_raw = engine.load_dataset(2025, vehicle_mode=vehicle_mode)
        elif dataset_key == "2026":
            df_raw = engine.load_dataset(2026, vehicle_mode=vehicle_mode)
        else:
            df_raw = engine.load_all_datasets(vehicle_mode=vehicle_mode)
    except Exception as e:
        logger.exception(f"DatasetEngine error: {e}")
        return pd.DataFrame()

    df = _run_pipeline(df_raw)
    df.attrs["dataset"]      = dataset_key
    df.attrs["vehicle_mode"] = vehicle_mode
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> tuple[str, str, str]:
    st.sidebar.title("⚡ ETIL")
    st.sidebar.markdown("**EV Transition Intelligence Lab**")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Dataset")
    dataset_option = st.sidebar.radio(
        "",
        ["IEA 2026", "IEA 2025", "Combined (2025+2026)"],
        index=0,
        key="dataset_radio",
    )
    dataset_key = {
        "IEA 2026":              "2026",
        "IEA 2025":              "2025",
        "Combined (2025+2026)":  "combined",
    }[dataset_option]

    st.sidebar.markdown("### Vehicle mode")
    vehicle_option = st.sidebar.selectbox(
        "",
        ["Cars", "2 and 3 wheelers", "Total (all modes)"],
        index=0,
        key="vehicle_select",
    )
    vehicle_mode = {
        "Cars":                "Cars",
        "2 and 3 wheelers":    "2 and 3 wheelers",
        "Total (all modes)":   "Total",
    }[vehicle_option]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    pages = {
        "🌍 Global Monitor":   "streamlit_pages.01_Global_Monitor",
        "🔬 Country Lab":      "streamlit_pages.02_Country_Lab",
        "🧠 Diffusion Engine": "streamlit_pages.03_Diffusion_Engine",
        "🔄 Regime Detection": "streamlit_pages.04_Regime_Detection",
        "📊 Forecast Lab":     "streamlit_pages.05_Forecast_Lab",
        "🏆 EVQTI Ranking":    "streamlit_pages.06_EVQTI_Ranking",
        "📰 EV Pulse":         "streamlit_pages.07_EV_Pulse",
    }
    selection = st.sidebar.radio("", list(pages.keys()), key="nav_radio")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Dataset: {dataset_option}")
    st.sidebar.caption(f"Mode: {vehicle_option}")
    st.sidebar.caption("Source: IEA Global EV Outlook")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Carlos Eduardo** · Quant Data Scientist  \n"
        "[LinkedIn](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · "
        "[GitHub](https://github.com/edu-moraess)"
    )

    return dataset_key, vehicle_mode, pages[selection]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dataset_key, vehicle_mode, module_name = _sidebar()

    df = get_data(dataset_key, vehicle_mode)
    if df is None or df.empty:
        st.error(
            "No data loaded. Place the IEA CSV files in `data/raw/` and reload."
        )
        return

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
 