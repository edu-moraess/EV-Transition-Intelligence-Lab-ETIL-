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
# Pipeline de dados
# ==========================================================

def load_and_process_data(vehicle_mode: str = "Cars") -> pd.DataFrame:
    """
    Pipeline ETIL usando IEA GEVO 2026.
    """

    try:
        loader = DataLoader()

        # O DataLoader tenta ev_data.csv e,
        # como ele foi removido, usa IEA 2026 automaticamente.
        df_raw = loader.load_from_csv("ev_data.csv")

        logger.info(
            f"Dados carregados: {len(df_raw)} registros"
        )

    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


    # Filtro de veículo
    if "mode" in df_raw.columns and vehicle_mode != "Total":
        df_raw = (
            df_raw[df_raw["mode"] == vehicle_mode]
            .copy()
        )


    # Limpeza
    cleaner = DataCleaner(df_raw)
    df_clean = cleaner.clean()


    # Transformações
    transformer = DataTransformer(df_clean)
    df_trans = transformer.add_log_transform(
        ["ev_sales"]
    )


    # Engenharia de features
    engineer = FeatureEngineer(df_trans)
    df_final = engineer.engineer()


    # Usa o EV stock share como penetração principal
    if "ev_stock_share" in df_final.columns:

        mask = (
            df_final["ev_stock_share"].notna()
            &
            (df_final["ev_stock_share"] > 0)
        )

        df_final.loc[
            mask,
            "penetration"
        ] = df_final.loc[
            mask,
            "ev_stock_share"
        ]


    logger.info(
        f"Pipeline concluído: {len(df_final)} registros finais."
    )


    df_final.attrs["vehicle_mode"] = vehicle_mode

    return df_final


# ==========================================================
# Aplicação principal
# ==========================================================

def main():

    st.sidebar.markdown("## ⚡ ETIL v2.1")
    st.sidebar.markdown(
        "**EV Transition Intelligence Lab**"
    )
    st.sidebar.markdown("---")


    vehicle_option = st.sidebar.selectbox(
        "Tipo de Veículo",
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


    pages = {
        "🌍 Global Monitor":
            "streamlit_pages.01_Global_Monitor",

        "🔬 Country Lab":
            "streamlit_pages.02_Country_Lab",

        "🧠 Diffusion Engine":
            "streamlit_pages.03_Diffusion_Engine",

        "🔄 Regime Detection":
            "streamlit_pages.04_Regime_Detection",

        "📊 Forecast Lab":
            "streamlit_pages.05_Forecast_Lab",

        "🏆 EVQTI Ranking":
            "streamlit_pages.06_EVQTI_Ranking",
    }


    selection = st.sidebar.radio(
        "Navegação",
        list(pages.keys())
    )


    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Fonte: IEA Global EV Outlook 2026"
    )
    st.sidebar.caption(
        f"Modo ativo: {vehicle_option}"
    )


    @st.cache_data(ttl=300)
    def get_data(mode):
        return load_and_process_data(mode)


    df_processed = get_data(vehicle_mode)


    if df_processed.empty:
        st.error(
            "Nenhum dado carregado."
        )
        return


    try:
        module = importlib.import_module(
            pages[selection]
        )

        if hasattr(module, "render"):
            module.render(df_processed)

        else:
            st.error(
                "Página sem função render()."
            )

    except Exception as e:

        import traceback

        st.error(
            f"Erro ao carregar página: {e}"
        )

        st.code(
            traceback.format_exc()
        )


if __name__ == "__main__":
    main()