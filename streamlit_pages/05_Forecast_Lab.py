# streamlit_pages/05_Forecast_Lab.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from ingestion.loader import DataLoader
from preprocessing.cleaning import DataCleaner
from preprocessing.features import FeatureEngineer
from models.forecasting.gaussian_process import GaussianProcessForecast
from models.forecasting.monte_carlo import MonteCarloSimulator

st.title("🔮 Forecast Lab")

loader = DataLoader()
df = loader.load_from_csv("EV_Data_Explorer_2026_GEVO_clean.csv")
df = DataCleaner(df).clean()
df = FeatureEngineer(df).engineer()

countries = sorted(df["country"].unique())
selected_country = st.selectbox("Selecione um país:", countries)

df_country = df[df["country"] == selected_country].sort_values("year")
y = df_country["ev_sales"].values
t = np.arange(len(y))

# Gaussian Process
gp = GaussianProcessForecast()
gp.fit(t, y)
t_future = np.arange(len(y) + 10)
y_pred, y_std = gp.predict(t_future)

# Monte Carlo
mc = MonteCarloSimulator(y)
simulations = mc.run(n_simulations=1000, horizon=10)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_country["year"], y=y, mode="markers", name="Dados"))
fig.add_trace(go.Scatter(x=df_country["year"].iloc[0] + t_future, y=y_pred, name="GP Forecast"))
fig.add_trace(go.Scatter(x=df_country["year"].iloc[0] + t_future, y=y_pred + 2*y_std, name="GP +2σ", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=df_country["year"].iloc[0] + t_future, y=y_pred - 2*y_std, name="GP -2σ", line=dict(dash="dot")))

st.plotly_chart(fig, use_container_width=True)

st.subheader("Monte Carlo Simulações")
st.line_chart(simulations.T)
