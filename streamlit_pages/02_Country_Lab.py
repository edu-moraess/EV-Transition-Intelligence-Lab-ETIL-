# streamlit_pages/02_Country_Lab.py
import streamlit as st
import pandas as pd
import plotly.express as px
from ingestion.loader import DataLoader
from preprocessing.cleaning import DataCleaner
from preprocessing.transformation import DataTransformer
from preprocessing.features import FeatureEngineer

st.title("🇨🇴 Country Lab")

# Carregar dados
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

# Seleção de país
countries = sorted(df_features["country"].unique())
selected_country = st.selectbox("Selecione um país:", countries)

df_country = df_features[df_features["country"] == selected_country]

# Mostrar dados
st.subheader(f"Dados para {selected_country}")
st.dataframe(df_country)

# Métricas principais
st.subheader("📊 Métricas principais")
col1, col2, col3 = st.columns(3)
col1.metric("Total EV Sales", f"{df_country['ev_sales'].sum():,.0f}")
col2.metric("CAGR", f"{df_country['cagr'].iloc[0]:.2f}%")
col3.metric("Penetração média", f"{df_country['penetration'].mean():.2f}%" if "penetration" in df_country else "N/A")

# Gráfico de vendas EV
st.subheader("📈 Evolução das vendas de EV")
fig = px.line(df_country, x="year", y="ev_sales", markers=True, title=f"EV Sales - {selected_country}")
st.plotly_chart(fig, use_container_width=True)

# Gráfico de penetração
if "penetration" in df_country:
    st.subheader("⚡ Penetração de EV")
    fig2 = px.line(df_country, x="year", y="penetration", markers=True, title=f"EV Penetration - {selected_country}")
    st.plotly_chart(fig2, use_container_width=True)
