# streamlit_pages/01_Global_Monitor.py
import streamlit as st
import pandas as pd
from ingestion.loader import DataLoader
from preprocessing.cleaning import DataCleaner
from preprocessing.transformation import DataTransformer
from preprocessing.features import FeatureEngineer

st.title("🌍 Global EV Monitor")

# Carregar dados (exemplo: CSV na pasta data/raw)
loader = DataLoader()
try:
    df = loader.load_from_csv("EV_Data_Explorer_2026_GEVO_clean.csv")
except FileNotFoundError:
    st.warning("Arquivo CSV não encontrado em data/raw/.")
    st.stop()

# Limpeza e features
cleaner = DataCleaner(df)
df_clean = cleaner.clean()

transformer = DataTransformer(df_clean)
df_transformed = transformer.add_log_transform(["ev_sales", "total_sales"])

engineer = FeatureEngineer(df_transformed)
df_features = engineer.engineer()

# Mostrar dados
st.subheader("Dados tratados")
st.dataframe(df_features.head(20))

# Estatísticas globais
st.subheader("Estatísticas globais")
st.metric("Total EV Sales", f"{df_features['ev_sales'].sum():,.0f}")
st.metric("Média CAGR", f"{df_features['cagr'].mean():.2f}%")
