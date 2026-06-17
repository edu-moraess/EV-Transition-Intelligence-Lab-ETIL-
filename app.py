import streamlit as st
import pandas as pd
import importlib

from ingestion.dataset_engine import DatasetEngine
from preprocessing.cleaning import DataCleaner
from preprocessing.transformation import DataTransformer
from preprocessing.features import FeatureEngineer
from utils.logger import setup_logger


logger = setup_logger(__name__)


# ==========================================================
# CONFIG STREAMLIT
# ==========================================================

st.set_page_config(
    page_title="EV Quant Research Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# DESIGN SYSTEM
# ==========================================================

st.markdown("""
<style>

[data-testid="stSidebar"] {
    background: #0d1117;
}

[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 4px 0;
}

.metric-label {
    color: #8b949e;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.metric-value {
    color: #e6edf3;
    font-size: 1.6rem;
    font-weight: 700;
}

h1, h2, h3 {
    color: #e6edf3 !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# PIPELINE ETIL
# ==========================================================

def load_and_process_data(vehicle_mode: str, dataset_mode: str) -> pd.DataFrame:

    try:
        engine = DatasetEngine(vehicle_mode)
        df_raw = engine.load(dataset_mode)

        logger.info(
            f"RAW: {len(df_raw)} linhas | {df_raw['country'].nunique()} países"
        )

    except Exception as e:
        logger.exception(f"Erro DatasetEngine: {e}")
        return pd.DataFrame()

    # ============================
    # CLEANING
    # ============================

    df_clean = DataCleaner(df_raw).clean()

    # ============================
    # TRANSFORMATION
    # ============================

    df_trans = DataTransformer(df_clean).add_log_transform(["ev_sales"])

    # ============================
    # FEATURES
    # ============================

    df_final = FeatureEngineer(df_trans).engineer()

    # ============================
    # FIX PENETRATION
    # ============================

    if "ev_stock_share" in df_final.columns:

        mask = (
            df_final["ev_stock_share"].notna()
            & (df_final["ev_stock_share"] > 0)
        )

        df_final.loc[mask, "penetration"] = df_final.loc[mask, "ev_stock_share"]

    df_final.attrs["vehicle_mode"] = vehicle_mode
    df_final.attrs["dataset_mode"] = dataset_mode

    return df_final


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ===============================
    # SIDEBAR HEADER
    # ===============================

    st.sidebar.markdown(
        """
        # EV Quant Research Platform  
        ### Global Transition Analytics  
        ---
        """,
        unsafe_allow_html=True
    )

    # ===============================
    # MODE VEHICLE
    # ===============================

    vehicle_option = st.sidebar.selectbox(
        "Modo de Análise",
        [
            "Cars (Passageiros)",
            "2 and 3 wheelers",
            "Total (Todos)"
        ],
        index=0
    )

    vehicle_mode = {
        "Cars (Passageiros)": "Cars",
        "2 and 3 wheelers": "2 and 3 wheelers",
        "Total (Todos)": "Total"
    }[vehicle_option]

    # ===============================
    # DATASET MODE (NOVO)
    # ===============================

    dataset_option = st.sidebar.selectbox(
        "Dataset",
        [
            "2025",
            "2026",
            "Combined"
        ],
        index=2
    )

    # ===============================
    # NAVIGATION
    # ===============================

    pages = {
        "🌍 Global Monitor": "streamlit_pages.01_Global_Monitor",
        "🔬 Country Lab": "streamlit_pages.02_Country_Lab",
        "🧠 Diffusion Engine": "streamlit_pages.03_Diffusion_Engine",
        "🔄 Regime Detection": "streamlit_pages.04_Regime_Detection",
        "📊 Forecast Lab": "streamlit_pages.05_Forecast_Lab",
        "🏆 EVQTI Ranking": "streamlit_pages.06_EVQTI_Ranking",
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navegação")

    selection = st.sidebar.radio("", list(pages.keys()))

    # ===============================
    # INFO
    # ===============================

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        f"""
        **Fonte:** IEA Global EV Outlook 2026  
        **Modo:** {vehicle_option}  
        **Dataset:** {dataset_option}
        """
    )

    # ===============================
    # SIGNATURE
    # ===============================

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        """
        **Eduardo Moraes**  
        Quant Systems Dev | Data Scientist
        """
    )

    # ===============================
    # CACHE
    # ===============================

    @st.cache_data(ttl=300, show_spinner="Processando dados IEA...")
    def get_data(mode, dataset):
        return load_and_process_data(mode, dataset)

    df_processed = get_data(vehicle_mode, dataset_option)

    if df_processed.empty:
        st.error("Nenhum dado foi carregado pelo pipeline.")
        return

    # ===============================
    # LOAD PAGE
    # ===============================

    try:
        module = importlib.import_module(pages[selection])

        if hasattr(module, "render"):
            module.render(df_processed)
        else:
            st.error("Página não possui render().")

    except Exception as e:
        import traceback
        st.error(f"Erro ao carregar página: {e}")
        st.code(traceback.format_exc())


# ==========================================================
# EXEC
# ==========================================================

if __name__ == "__main__":
    main()
