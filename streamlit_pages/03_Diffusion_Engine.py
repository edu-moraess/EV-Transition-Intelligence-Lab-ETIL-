# streamlit_pages/03_Diffusion_Engine.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from ingestion.loader import DataLoader
from preprocessing.cleaning import DataCleaner
from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel

st.title("📈 Diffusion Engine")

loader = DataLoader()
df = loader.load_from_csv("EV_Data_Explorer_2026_GEVO_clean.csv")
df = DataCleaner(df).clean()

countries = sorted(df["country"].unique())
selected_country = st.selectbox("Selecione um país:", countries)

df_country = df[df["country"] == selected_country].sort_values("year")
t = np.arange(len(df_country))
S = df_country["ev_sales"].cumsum().values

# Ajustar modelos
bass = BassDiffusion()
bass.fit(t, S)
logistic = LogisticGrowth()
logistic.fit(t, S)
gompertz = GompertzModel()
gompertz.fit(t, S)

# Previsões
t_future = np.arange(len(df_country) + 10)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_country["year"], y=S, mode="markers", name="Dados"))
fig.add_trace(go.Scatter(x=df_country["year"].iloc[0] + t_future, y=bass.predict(t_future), name="Bass"))
fig.add_trace(go.Scatter(x=df_country["year"].iloc[0] + t_future, y=logistic.predict(t_future), name="Logístico"))
fig.add_trace(go.Scatter(x=df_country["year"].iloc[0] + t_future, y=gompertz.predict(t_future), name="Gompertz"))

st.plotly_chart(fig, use_container_width=True)
