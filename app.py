import streamlit as st
import pandas as pd
import importlib

from ingestion.loader import DataLoader
from preprocessing.cleaning import DataCleaner
from preprocessing.transformation import DataTransformer
from preprocessing.features import FeatureEngineer
from utils.logger import setup_logger


logger = setup_logger(__name__)


# ==========================================================
# Configuração Streamlit
# ==========================================================

st.set_page_config(
    page_title="EV Transition Intelligence Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# Design System
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

.delta-up {
    color: #3fb950;
}

.delta-down {
    color: #f85149;
}

h1, h2, h3 {
    color: #e6edf3 !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# Pipeline ETIL
# ==========================================================

def load_and_process_data(vehicle_mode: str = "Cars") -> pd.DataFrame:

    try:
        loader = DataLoader()

        df_raw = loader.load_from_csv(vehicle_mode=vehicle_mode)

        logger.info(
            f"RAW: {len(df_raw)} linhas | {df_raw['country'].nunique()} países"
        )

    except Exception as e:
        logger.exception(f"Erro no DataLoader: {e}")
        return pd.DataFrame()

    # ============================
    # Cleaning
    # ============================

    df_clean = DataCleaner(df_raw).clean()

    logger.info(
        f"CLEAN: {len(df_clean)} linhas | {df_clean['country'].nunique()} países"
    )

    # ============================
    # Transformation
    # ============================

    transformer = DataTransformer(df_clean)

    df_trans = transformer.add_log_transform(["ev_sales"])

    logger.info(
        f"TRANSFORM: {len(df_trans)} linhas | {df_trans['country'].nunique()} países"
    )

    # ============================
    # Feature Engineering
    # ============================

    engineer = FeatureEngineer(df_trans)

    df_final = engineer.engineer()

    logger.info(
        f"FEATURES: {len(df_final)} linhas | {df_final['country'].nunique()} países"
    )

    # ============================
    # Penetration fix
    # ============================

    if "ev_stock_share" in df_final.columns:

        mask = (
            df_final["ev_stock_share"].notna()
            & (df_final["ev_stock_share"] > 0)
        )

        df_final.loc[mask, "penetration"] = df_final.loc[mask, "ev_stock_share"]

    df_final.attrs["vehicle_mode"] = vehicle_mode

    logger.info(
        "Pipeline ETIL concluído"
    )

    return df_final


# ==========================================================
# Aplicação principal
# ==========================================================

def main():

    # ===============================
    # SIDEBAR HEADER (NOVO)
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
    # MODE SELECTION
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

    selection = st.sidebar.radio(
        "",
        list(pages.keys())
    )

    # ===============================
    # INFO DINÂMICA
    # ===============================

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        f"""
        **Fonte:** IEA Global EV Outlook 2026  
        **Modo ativo:** {vehicle_option}
        """
    )

    # ===============================
    # ASSINATURA
    # ===============================

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        """
        **Eduardo Moraes**  
        Quant Systems Dev | Data Scientist
        """
    )

    # ===============================
    # CACHE DATA
    # ===============================

    @st.cache_data(ttl=300, show_spinner="Processando dados IEA...")
    def get_data(mode):
        return load_and_process_data(mode)

    df_processed = get_data(vehicle_mode)

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
            st.error("Página selecionada não possui render().")

    except Exception as e:
        import traceback

        st.error(f"Erro ao carregar página: {e}")
        st.code(traceback.format_exc())


# ==========================================================
# EXEC
# ==========================================================

if __name__ == "__main__":
    main()
