# streamlit_pages/04_Regime_Detection.py
"""
Página: Regime Detection - HMM e changepoints.
"""
import streamlit as st
import pandas as pd
import numpy as np
from models.regimes import HMMRegimeDetector, BayesianChangePoint
from visualization.regime_dashboard import RegimeDashboard

def render(df_processed):
    st.title("🔄 Regime Detection")
    countries = sorted(df_processed["country"].unique())
    selected = st.selectbox("Selecione um país", countries)

    if selected:
        data = df_processed[df_processed["country"] == selected].sort_values("year")
        if len(data) < 5:
            st.warning("Dados insuficientes para detecção de regimes.")
            return

        # Preparar features
        features = []
        if "penetration" in data.columns and data["penetration"].notna().any():
            features.append(data["penetration"].fillna(0).values.reshape(-1, 1))
        if "growth_rate" in data.columns and data["growth_rate"].notna().any():
            features.append(data["growth_rate"].fillna(0).values.reshape(-1, 1))
        if not features:
            st.error("Features 'penetration' ou 'growth_rate' não disponíveis.")
            return
        X = np.hstack(features)

        # HMM
        hmm = HMMRegimeDetector(n_states=4)
        try:
            hmm.fit(X)
            states, probs = hmm.predict_regime(X)
            st.subheader("Regimes detectados (HMM)")
            regime_names = [hmm.get_regime_name(s) for s in states]
            st.write("Estados:", regime_names)
            # Plot
            fig = RegimeDashboard.plot_regimes(data["ev_sales"].values, states, probs, hmm.REGIME_NAMES)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"HMM falhou: {e}")

        # Changepoint
        st.subheader("Bayesian Change Point")
        cp = BayesianChangePoint()
        cp_points = cp.detect(data["ev_sales"].values)
        st.write("Pontos de mudança nos índices:", cp_points)