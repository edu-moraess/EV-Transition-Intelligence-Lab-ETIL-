# streamlit_pages/04_Regime_Detection.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models.regimes import HMMRegimeDetector, BayesianChangePoint

_BG="#161b22"; _GRID="#21262d"; _TEXT="#e6edf3"
REGIME_COLORS = {0:"#8b949e", 1:"#388bfd", 2:"#3fb950", 3:"#d2a8ff"}
REGIME_NAMES  = {0:"Emerging", 1:"Expansion", 2:"Acceleration", 3:"Maturity"}


def render(df: pd.DataFrame):
    st.title("🔄 Regime Detection")
    st.markdown("Detecção de regimes de adoção via Gaussian Mixture Model + Bayesian Changepoint")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    col_s, col_n = st.columns([3,1])
    with col_s:
        selected = st.selectbox("País", countries,
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

    with st.spinner("Ajustando GMM..."):
        detector = HMMRegimeDetector(n_states=n_states)
        try:
            detector.fit(X)
            states, probs = detector.predict_regime(X)
        except Exception as e:
            st.error(f"Modelo falhou: {e}"); return

    regime_names_map = {i: ["Emerging","Expansion","Acceleration","Maturity","Peak"][i]
                        for i in range(n_states)}
    reg_names = [regime_names_map[s] for s in states]
    reg_colors_list = [list(REGIME_COLORS.values())[s % len(REGIME_COLORS)] for s in states]

    # Plot principal
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35],
                        subplot_titles=["Vendas EV + Regimes", "Probabilidade por Regime"])

    # Série
    fig.add_trace(go.Scatter(
        x=data["year"].values, y=data["ev_sales"].values/1e3,
        mode="lines", name="Vendas EV (k)",
        line=dict(color="#e6edf3", width=2)
    ), row=1, col=1)

    # Pontos coloridos por regime
    for s in range(n_states):
        mask = np.array(states) == s
        if mask.any():
            color = list(REGIME_COLORS.values())[s % len(REGIME_COLORS)]
            fig.add_trace(go.Scatter(
                x=data["year"].values[mask], y=data["ev_sales"].values[mask]/1e3,
                mode="markers", name=regime_names_map[s],
                marker=dict(size=10, color=color, line=dict(width=1.5, color="#0d1117"))
            ), row=1, col=1)

    # Probabilidades empilhadas
    for s in range(probs.shape[1]):
        color = list(REGIME_COLORS.values())[s % len(REGIME_COLORS)]
        fig.add_trace(go.Scatter(
            x=data["year"].values, y=probs[:, s],
            mode="lines", name=regime_names_map[s],
            fill="tonexty" if s > 0 else "tozeroy",
            line=dict(color=color, width=0.5),
            fillcolor=color.replace(")", ",0.3)").replace("rgb","rgba") if "rgb" in color else color,
            showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_TEXT),
        height=520, margin=dict(l=0,r=0,t=40,b=0),
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d"),
        xaxis2=dict(title="Ano", gridcolor=_GRID),
        yaxis=dict(title="Mil unidades", gridcolor=_GRID),
        yaxis2=dict(title="Probabilidade", gridcolor=_GRID, range=[0,1])
    )
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT
    st.plotly_chart(fig, use_container_width=True)

    # Tabela regimes
    st.markdown("#### Sequência de regimes por ano")
    reg_df = pd.DataFrame({
        "Ano": data["year"].values,
        "Regime": reg_names,
        "Vendas EV (k)": (data["ev_sales"].values/1e3).round(1),
        "Crescimento YoY (%)": data["growth_rate"].fillna(0).round(1).values
    })
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Bayesian Changepoint
    st.subheader("📍 Bayesian Change Point Detection")
    cp = BayesianChangePoint()
    cp_points = cp.detect(data["ev_sales"].values)
    if cp_points:
        cp_years = [int(data["year"].values[i]) for i in cp_points if i < len(data)]
        st.success(f"Pontos de mudança detectados nos anos: **{', '.join(map(str, cp_years))}**")
        fig_cp = go.Figure()
        fig_cp.add_trace(go.Scatter(
            x=data["year"].values, y=data["ev_sales"].values/1e3,
            mode="lines+markers", name="Vendas EV", line=dict(color="#388bfd")
        ))
        for yr in cp_years:
            fig_cp.add_vline(x=yr, line_dash="dash", line_color="#f0883e",
                             annotation_text=str(yr), annotation_font_color="#f0883e")
        fig_cp.update_layout(
            paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_TEXT),
            xaxis=dict(title="Ano", gridcolor=_GRID),
            yaxis=dict(title="Mil unidades", gridcolor=_GRID),
            height=300, margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig_cp, use_container_width=True)
    else:
        st.info("Nenhum ponto de mudança estrutural detectado.")
