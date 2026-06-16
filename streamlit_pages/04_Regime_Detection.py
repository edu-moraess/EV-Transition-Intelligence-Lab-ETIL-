# streamlit_pages/04_Regime_Detection.py
import streamlit as st
import numpy as np
import plotly.express as px
from ingestion.loader import DataLoader
from preprocessing.cleaning import DataCleaner
from preprocessing.features import FeatureEngineer
from models.regimes import HMMRegimeDetector

st.title("🔀 Regime Detection")

loader = DataLoader()
df = loader.load_from_csv("EV_Data_Explorer_2026_GEVO_clean.csv")
df = DataCleaner(df).clean()
df = FeatureEngineer(df).engineer()

countries = sorted(df["country"].unique())
selected_country = st.selectbox("Selecione um país:", countries)

df_country = df[df["country"] == selected_country].sort_values("year")

X = df_country[["growth_rate", "momentum_3y", "volatility"]].fillna(0).values
hmm = HMMRegimeDetector()
hmm.fit(X)
states, probs = hmm.predict_regime(X)

df_country["regime"] = [hmm.get_regime_name(s) for s in states]

st.subheader("Regimes detectados")
st.dataframe(df_country[["year", "ev_sales", "growth_rate", "regime"]])

fig = px.scatter(df_country, x="year", y="ev_sales", color="regime", title=f"Regimes - {selected_country}")
st.plotly_chart(fig, use_container_width=True)
