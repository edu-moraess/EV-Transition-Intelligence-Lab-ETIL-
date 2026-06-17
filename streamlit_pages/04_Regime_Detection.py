# streamlit_pages/04_Regime_Detection.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from models.regimes import HMMRegimeDetector, BayesianChangePoint

REGIME_NAMES = {0: "Emerging", 1: "Expansion", 2: "Acceleration", 3: "Maturity", 4: "Peak"}


def render(df: pd.DataFrame):
    st.title("🔄 Regime Detection")
    st.caption("Gaussian Mixture Model (proxy HMM) + Bayesian Changepoint Detection")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    col_s, col_n = st.columns([3, 1])
    with col_s:
        selected  = st.selectbox("País", countries,
                                  index=countries.index("Norway") if "Norway" in countries else 0)
    with col_n:
        n_states = st.selectbox("Nº de regimes", [3, 4, 5], index=1)

    data = df[df["country"] == selected].sort_values("year").reset_index(drop=True)
    if len(data) < 5:
        st.warning("Dados insuficientes (mín. 5 anos)."); return

    # Features
    feat_cols = []
    if "ev_stock_share" in data.columns and data["ev_stock_share"].notna().any():
        feat_cols.append("ev_stock_share")
    elif "penetration" in data.columns and data["penetration"].notna().any():
        feat_cols.append("penetration")
    if "growth_rate" in data.columns and data["growth_rate"].notna().any():
        feat_cols.append("growth_rate")

    if not feat_cols:
        st.error("Features insuficientes."); return

    X = np.column_stack([data[c].fillna(0).values for c in feat_cols])
    regime_name_map = {i: list(REGIME_NAMES.values())[i] for i in range(n_states)}

    with st.spinner("Ajustando GMM..."):
        detector = HMMRegimeDetector(n_states=n_states)
        try:
            detector.fit(X)
            states, probs = detector.predict_regime(X)
        except Exception as e:
            st.error(f"Modelo falhou: {e}"); return

    reg_names_list = [regime_name_map[s] for s in states]

    # Plot principal
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.65, 0.35],
                         subplot_titles=["Vendas EV + Regimes", "Probabilidade por Regime"])

    fig.add_trace(go.Scatter(x=data["year"].values, y=data["ev_sales"].values/1e3,
                               mode="lines", name="Vendas EV (k)", line=dict(width=2)), row=1, col=1)

    palette = px.colors.qualitative.Plotly
    for s in range(n_states):
        mask = np.array(states) == s
        if mask.any():
            fig.add_trace(go.Scatter(
                x=data["year"].values[mask], y=data["ev_sales"].values[mask]/1e3,
                mode="markers", name=regime_name_map[s],
                marker=dict(size=10, color=palette[s % len(palette)],
                            line=dict(width=1.5, color="white"))
            ), row=1, col=1)

    for s in range(probs.shape[1]):
        fig.add_trace(go.Scatter(
            x=data["year"].values, y=probs[:, s],
            mode="lines", name=regime_name_map[s],
            fill="tonexty" if s > 0 else "tozeroy",
            line=dict(color=palette[s % len(palette)], width=0.5),
            showlegend=False
        ), row=2, col=1)

    fig.update_layout(height=500, margin=dict(l=0,r=0,t=40,b=0),
                       yaxis=dict(title="k unidades"),
                       xaxis2=dict(title="Ano"), yaxis2=dict(title="Probabilidade", range=[0,1]))
    st.plotly_chart(fig, use_container_width=True)

    reg_df = pd.DataFrame({
        "Ano": data["year"].values, "Regime": reg_names_list,
        "Vendas (k)": (data["ev_sales"].values/1e3).round(1),
        "YoY (%)": data["growth_rate"].fillna(0).round(1).values
    })
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📍 Bayesian Change Point Detection")
    cp     = BayesianChangePoint()
    cp_pts = cp.detect(data["ev_sales"].values)
    if cp_pts:
        cp_years = [int(data["year"].values[i]) for i in cp_pts if i < len(data)]
        st.success(f"Pontos de mudança: **{', '.join(map(str, cp_years))}**")
        fig_cp = px.line(data, x="year", y="ev_sales",
                          labels={"ev_sales": "Vendas EV", "year": "Ano"})
        for yr in cp_years:
            fig_cp.add_vline(x=yr, line_dash="dash", annotation_text=str(yr))
        fig_cp.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_cp, use_container_width=True)
    else:
        st.info("Nenhum ponto de mudança estrutural detectado.")
