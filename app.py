# app.py
"""
EV Transition Intelligence Lab — app principal.

Usa exclusivamente o tema nativo do Streamlit (config.toml).
Nenhuma cor fixa no código Python.
"""
import streamlit as st
import pandas as pd
import numpy as np
import importlib
from pathlib import Path
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

# ── Pipeline ─────────────────────────────────────────────────────────────────

def _run_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Executa limpeza, transformação e feature engineering."""
    df = DataCleaner(df_raw).clean()
    df = DataTransformer(df).add_log_transform(["ev_sales"])
    df = FeatureEngineer(df).engineer()

    # Penetration fix: ev_stock_share é a métrica real de penetração
    if "ev_stock_share" in df.columns:
        mask = df["ev_stock_share"].notna() & (df["ev_stock_share"] > 0)
        df.loc[mask, "penetration"] = df.loc[mask, "ev_stock_share"]

    # Propagar ev_sales_share do raw (não processado pelo FeatureEngineer)
    if "ev_sales_share" in df_raw.columns:
        share_raw = df_raw[["country", "year", "ev_sales_share"]].drop_duplicates()
        df = df.merge(share_raw, on=["country", "year"], how="left", suffixes=("", "_raw"))
        if "ev_sales_share_raw" in df.columns:
            if "ev_sales_share" not in df.columns:
                df["ev_sales_share"] = df["ev_sales_share_raw"]
            df.drop(columns=["ev_sales_share_raw"], inplace=True, errors="ignore")

    return df


@st.cache_data(ttl=600, show_spinner="Carregando dados IEA...")
def get_data(dataset_key: str, vehicle_mode: str) -> pd.DataFrame:
    """Carrega e processa dados com cache por (dataset, modo)."""
    engine = DatasetEngine()
    try:
        if dataset_key == "2025":
            df_raw = engine.load_dataset(2025, vehicle_mode=vehicle_mode)
        elif dataset_key == "2026":
            df_raw = engine.load_dataset(2026, vehicle_mode=vehicle_mode)
        else:
            df_raw = engine.load_all_datasets(vehicle_mode=vehicle_mode)
    except Exception as e:
        logger.exception(f"Erro no DatasetEngine: {e}")
        return pd.DataFrame()

    logger.info(f"RAW [{dataset_key}]: {df_raw.shape[0]} linhas | {df_raw['country'].nunique()} países")
    df = _run_pipeline(df_raw)
    df.attrs["dataset"] = dataset_key
    df.attrs["vehicle_mode"] = vehicle_mode
    return df


# ── App principal ─────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.title("⚡ ETIL")
    st.sidebar.markdown("**EV Transition Intelligence Lab**")
    st.sidebar.markdown("---")

    # Dataset selector
    st.sidebar.markdown("### Dataset IEA")
    dataset_option = st.sidebar.radio(
        "",
        ["IEA 2026", "IEA 2025", "Combinado (2025+2026)"],
        index=0,
        key="dataset_radio",
    )
    dataset_key = {"IEA 2026": "2026", "IEA 2025": "2025", "Combinado (2025+2026)": "combined"}[dataset_option]

    # Modo de veículo
    st.sidebar.markdown("### Modo de transporte")
    vehicle_option = st.sidebar.selectbox(
        "",
        ["Cars (Passageiros)", "2 and 3 wheelers", "Total (Todos)"],
        index=0,
        key="vehicle_select",
    )
    vehicle_mode = {
        "Cars (Passageiros)":  "Cars",
        "2 and 3 wheelers":    "2 and 3 wheelers",
        "Total (Todos)":       "Total",
    }[vehicle_option]

    # Navegação
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navegação")
    pages = {
        "🌍 Global Monitor":   "streamlit_pages.01_Global_Monitor",
        "🔬 Country Lab":      "streamlit_pages.02_Country_Lab",
        "🧠 Diffusion Engine": "streamlit_pages.03_Diffusion_Engine",
        "🔄 Regime Detection": "streamlit_pages.04_Regime_Detection",
        "📊 Forecast Lab":     "streamlit_pages.05_Forecast_Lab",
        "🏆 EVQTI Ranking":    "streamlit_pages.06_EVQTI_Ranking",
    }
    selection = st.sidebar.radio("", list(pages.keys()), key="nav_radio")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Dataset: {dataset_option}")
    st.sidebar.caption(f"Modo: {vehicle_option}")
    st.sidebar.caption("Fonte: IEA Global EV Outlook")
    st.sidebar.markdown("---")
    st.sidebar.caption("**Eduardo Moraes** · Quant Data Scientist")

    # ── Carregamento ──────────────────────────────────────────────────────
    df = get_data(dataset_key, vehicle_mode)

    if df is None or df.empty:
        st.error("Nenhum dado carregado. Verifique os arquivos IEA em data/raw/.")
        return

    # ── Renderização de página ────────────────────────────────────────────
    try:
        module = importlib.import_module(pages[selection])
        if hasattr(module, "render"):
            module.render(df)
        else:
            st.error(f"Módulo {pages[selection]} não possui função render().")
    except Exception as e:
        import traceback
        st.error(f"Erro ao carregar página: {e}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
