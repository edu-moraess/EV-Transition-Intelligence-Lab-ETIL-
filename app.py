# app.py
"""
Aplicação principal Streamlit.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from ingestion.loader import DataLoader
from preprocessing.cleaning import DataCleaner
from preprocessing.transformation import DataTransformer
from preprocessing.features import FeatureEngineer
from streamlit_pages import (
    GlobalMonitor,
    CountryLab,
    DiffusionEngine,
    RegimeDetection,
    ForecastLab,
    EVQTIRanking,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

st.set_page_config(page_title="EV Transition Intelligence Lab", layout="wide")

def load_and_process_data():
    """
    Carrega e processa os dados.
    """
    loader = DataLoader()
    # Tentar carregar CSV padrão
    try:
        df_raw = loader.load_from_csv("ev_data.csv")
    except FileNotFoundError:
        # Se não encontrar, carrega um exemplo dummy para demonstração
        st.warning("Arquivo 'ev_data.csv' não encontrado. Usando dados de exemplo.")
        df_raw = generate_dummy_data()

    # Limpeza
    cleaner = DataCleaner(df_raw)
    df_clean = cleaner.clean()

    # Transformação (opcional)
    transformer = DataTransformer(df_clean)
    df_trans = transformer.add_log_transform(["ev_sales"])

    # Feature engineering
    engineer = FeatureEngineer(df_trans)
    df_final = engineer.engineer()

    return df_final

def generate_dummy_data():
    """Gera dados sintéticos para demonstração."""
    countries = ["USA", "China", "Germany", "France", "UK", "Norway", "Japan", "India", "Brazil"]
    years = list(range(2015, 2026))
    data = []
    for country in countries:
        base = 1000 if country in ["USA", "China"] else 100
        for year in years:
            ev = base * (1.2 ** (year - 2015)) * (0.9 + 0.2 * np.random.rand())
            total = ev * (5 + 10 * np.random.rand())
            charging = ev * (0.01 + 0.02 * np.random.rand())
            pop = 50e6 if country in ["USA", "China"] else 10e6
            gdp = 2e12 if country in ["USA", "China"] else 0.5e12
            data.append([country, year, ev, total, charging, pop, gdp])
    df = pd.DataFrame(data, columns=["country", "year", "ev_sales", "total_sales", "charging_points", "population", "gdp"])
    return df

def main():
    st.sidebar.title("EV Transition Intelligence Lab")
    st.sidebar.markdown("---")
    pages = {
        "🌍 Global Monitor": GlobalMonitor,
        "🔬 Country Lab": CountryLab,
        "🧠 Diffusion Engine": DiffusionEngine,
        "🔄 Regime Detection": RegimeDetection,
        "📊 Forecast Lab": ForecastLab,
        "🏆 EVQTI Ranking": EVQTIRanking,
    }
    selection = st.sidebar.radio("Navegação", list(pages.keys()))

    # Carrega dados (cache para performance)
    @st.cache_data
    def get_data():
        return load_and_process_data()

    df_processed = get_data()
    if df_processed is None or df_processed.empty:
        st.error("Falha ao carregar dados.")
        return

    # Renderiza página selecionada
    pages[selection].render(df_processed)

if __name__ == "__main__":
    main()