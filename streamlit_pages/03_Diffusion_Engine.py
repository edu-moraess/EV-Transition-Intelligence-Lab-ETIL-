# streamlit_pages/03_Diffusion_Engine.py
"""
Página: Diffusion Engine - ajuste e visualização de modelos.
"""
import streamlit as st
import pandas as pd
import numpy as np
from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel
from visualization.diffusion_plot import DiffusionPlot

def render(df_processed):
    st.title("🧠 Diffusion Engine")
    countries = sorted(df_processed["country"].unique())
    selected = st.selectbox("Selecione um país", countries)

    if selected:
        data = df_processed[df_processed["country"] == selected].sort_values("year")
        t = np.arange(len(data))
        S = data["ev_sales"].cumsum().values

        # Ajusta modelos
        models = {
            "Bass": BassDiffusion(),
            "Logistic": LogisticGrowth(),
            "Gompertz": GompertzModel()
        }
        fitted_models = {}
        for name, model in models.items():
            try:
                model.fit(t, S)
                fitted_models[name] = model
            except:
                st.warning(f"{name} não convergiu para {selected}")

        if fitted_models:
            fig = DiffusionPlot.plot_models(t, S, fitted_models)
            st.plotly_chart(fig, use_container_width=True)