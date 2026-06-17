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

# ── Design system ────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d1117; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stRadio label { 
    font-size: 0.92rem; padding: 4px 0; 
}
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


def load_and_process_data():
    loader = DataLoader()
    try:
        df_raw = loader.load_from_csv("ev_data.csv")
    except Exception:
        df_raw = generate_dummy_data()
    df_clean  = DataCleaner(df_raw).clean()
    df_trans  = DataTransformer(df_clean).add_log_transform(["ev_sales"])
    df_final  = FeatureEngineer(df_trans).engineer()
    # Usar ev_stock_share como penetração real onde disponível
    if "ev_stock_share" in df_final.columns:
        mask = df_final["ev_stock_share"].notna() & (df_final["ev_stock_share"] > 0)
        df_final.loc[mask, "penetration"] = df_final.loc[mask, "ev_stock_share"]
    return df_final


def generate_dummy_data():
    np.random.seed(42)
    countries = ["USA","China","Germany","France","UK","Norway","Japan","India","Brazil"]
    years = list(range(2015, 2026))
    data = []
    for c in countries:
        base = 1000 if c in ["USA","China"] else 100
        for y in years:
            ev    = base*(1.2**(y-2015))*(0.9+0.2*np.random.rand())
            total = ev*(5+10*np.random.rand())
            chg   = ev*(0.01+0.02*np.random.rand())
            pop   = 50e6 if c in ["USA","China"] else 10e6
            gdp   = 2e12 if c in ["USA","China"] else 0.5e12
            data.append([c,y,ev,total,chg,pop,gdp])
    return pd.DataFrame(data, columns=["country","year","ev_sales","total_sales",
                                        "charging_points","population","gdp"])


def main():
    # Sidebar
    st.sidebar.markdown("## ⚡ ETIL")
    st.sidebar.markdown("**EV Transition Intelligence Lab**")
    st.sidebar.markdown("---")
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
    st.sidebar.caption("Modelo: GMM · GP · Bass · Monte Carlo")

    @st.cache_data(show_spinner="Carregando dados IEA...")
    def get_data():
        return load_and_process_data()

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
