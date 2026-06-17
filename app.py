# app.py
import streamlit as st
import pandas as pd
import numpy as np
import importlib
from ingestion.loader import DataLoader
from preprocessing.cleaning import DataCleaner
from preprocessing.transformation import DataTransformer
from preprocessing.features import FeatureEngineer
from utils.logger import setup_logger

logger = setup_logger(__name__)

st.set_page_config(
    page_title="EV Transition Intelligence Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design system
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d1117; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 16px 20px; margin: 4px 0;
}
.metric-label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; }
.metric-value { color: #e6edf3; font-size: 1.6rem; font-weight: 700; margin-top: 2px; }
.metric-delta { font-size: 0.8rem; margin-top: 4px; }
.delta-up   { color: #3fb950; }
.delta-down { color: #f85149; }
h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)


def load_and_process_data(vehicle_mode: str = "Cars"):
    """Pipeline com filtro por tipo de veículo"""
    loader = DataLoader()
    
    try:
        df_raw = loader.load_from_csv("ev_data.csv")
    except Exception:
        logger.info("Usando dados IEA 2026 com filtro")
        df_raw = loader._load_iea_gevo()  # força carregamento IEA

    # === FILTRO POR TIPO DE VEÍCULO ===
    if 'mode' in df_raw.columns:
        if vehicle_mode == "Cars":
            df_raw = df_raw[df_raw['mode'] == "Cars"]
        elif vehicle_mode == "2 and 3 wheelers":
            df_raw = df_raw[df_raw['mode'] == "2 and 3 wheelers"]
        # "Total" mantém tudo

    cleaner = DataCleaner(df_raw)
    df_clean = cleaner.clean()

    transformer = DataTransformer(df_clean)
    df_trans = transformer.add_log_transform(["ev_sales"])

    engineer = FeatureEngineer(df_trans)
    df_final = engineer.engineer()

    # Prioriza ev_stock_share como penetration
    if "ev_stock_share" in df_final.columns:
        mask = df_final["ev_stock_share"].notna() & (df_final["ev_stock_share"] > 0)
        df_final.loc[mask, "penetration"] = df_final.loc[mask, "ev_stock_share"]

    logger.info(f"Dados processados: {df_final.shape[0]} registros | Modo: {vehicle_mode}")
    return df_final


def main():
    st.sidebar.markdown("## ⚡ ETIL v2")
    st.sidebar.markdown("**EV Transition Intelligence Lab**")
    st.sidebar.markdown("---")

    # === FILTRO GLOBAL POR TIPO DE VEÍCULO ===
    vehicle_mode = st.sidebar.selectbox(
        "Tipo de Veículo",
        ["Cars (Passageiros)", "2 and 3 wheelers", "Total (Todos)"],
        index=0
    )
    vehicle_key = {
        "Cars (Passageiros)": "Cars",
        "2 and 3 wheelers": "2 and 3 wheelers",
        "Total (Todos)": "Total"
    }[vehicle_mode]

    pages = {
        "🌍 Global Monitor":   "streamlit_pages.01_Global_Monitor",
        "🔬 Country Lab":      "streamlit_pages.02_Country_Lab",
        "🧠 Diffusion Engine": "streamlit_pages.03_Diffusion_Engine",
        "🔄 Regime Detection": "streamlit_pages.04_Regime_Detection",
        "📊 Forecast Lab":     "streamlit_pages.05_Forecast_Lab",
        "🏆 EVQTI Ranking":    "streamlit_pages.06_EVQTI_Ranking",
    }
    selection = st.sidebar.radio("Navegação", list(pages.keys()))

    st.sidebar.markdown("---")
    st.sidebar.caption("Fonte: IEA Global EV Outlook 2026")
    st.sidebar.caption(f"Modo atual: **{vehicle_mode}**")

    @st.cache_data(show_spinner=f"Carregando dados IEA ({vehicle_mode})...")
    def get_data():
        return load_and_process_data(vehicle_key)

    df_processed = get_data()

    if df_processed is None or df_processed.empty:
        st.error("Falha ao carregar dados.")
        return

    module_name = pages[selection]
    try:
        page_module = importlib.import_module(module_name)
        if hasattr(page_module, "render"):
            page_module.render(df_processed)
        else:
            st.error(f"Módulo {module_name} sem função 'render'.")
    except Exception as e:
        st.error(f"Erro ao carregar a página: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()