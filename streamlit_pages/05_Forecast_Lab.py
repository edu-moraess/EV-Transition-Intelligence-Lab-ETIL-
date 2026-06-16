# streamlit_pages/05_Forecast_Lab.py
"""
Página: Forecast Lab - previsões e Monte Carlo.
"""
import streamlit as st
import pandas as pd
import numpy as np
from models.forecasting import BayesianGPForecast, MonteCarloSimulator
from visualization.forecast_dashboard import ForecastDashboard

def render(df_processed):
    st.title("📊 Forecast Lab")
    countries = sorted(df_processed["country"].unique())
    selected = st.selectbox("Selecione um país", countries)

    if selected:
        data = df_processed[df_processed["country"] == selected].sort_values("year")
        t_hist = np.arange(len(data)).reshape(-1, 1)
        y_hist = data["ev_sales"].values

        horizon = st.slider("Horizonte de previsão (anos)", 1, 20, 10)
        t_future = np.arange(len(data), len(data) + horizon).reshape(-1, 1)

        # GP
        gp = BayesianGPForecast()
        gp.fit(t_hist, y_hist)
        mean, std = gp.predict(t_future, return_std=True)

        # Monte Carlo
        mc = MonteCarloSimulator(n_simulations=2000)
        sims = mc.simulate_from_gp(gp, t_future)
        lower, median, upper = mc.get_percentiles(sims)

        # Plot
        fig = ForecastDashboard.plot_forecast(
            t_hist.flatten(), y_hist,
            t_future.flatten(), median, lower, upper
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cenários
        st.subheader("Cenários")
        scenario = st.selectbox("Escolha um cenário", ["base", "optimistic", "pessimistic"])
        adjusted = mc.scenario_adjust(sims, scenario)
        lower_a, median_a, upper_a = mc.get_percentiles(adjusted)
        fig2 = ForecastDashboard.plot_forecast(
            t_hist.flatten(), y_hist,
            t_future.flatten(), median_a, lower_a, upper_a,
            scenario_label=f"Cenário {scenario.capitalize()}"
        )
        st.plotly_chart(fig2, use_container_width=True)