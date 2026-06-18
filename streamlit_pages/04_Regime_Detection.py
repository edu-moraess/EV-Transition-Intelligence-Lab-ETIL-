# streamlit_pages/04_Regime_Detection.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models.regimes import HMMRegimeDetector, BayesianChangePoint
from visualization.theme import (
    base_layout, NAVY, NAVY_LIGHT, RED_DARK,
    GREEN_MED, AMBER, SLATE, BORDER
)

_REGIME_LABELS  = {0:"Emerging", 1:"Expansion", 2:"Acceleration", 3:"Maturity", 4:"Peak"}
_REGIME_COLORS  = {0:SLATE, 1:NAVY_LIGHT, 2:GREEN_MED, 3:AMBER, 4:RED_DARK}


def render(df: pd.DataFrame):
    st.title("🔄 Regime Detection")
    st.caption("Gaussian Mixture Model (proxy HMM) + Bayesian Changepoint Detection")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    col_s, col_n = st.columns([3, 1])
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
    regime_map = {i: _REGIME_LABELS.get(i, f"R{i}") for i in range(n_states)}
    color_map  = {i: _REGIME_COLORS.get(i, SLATE) for i in range(n_states)}

    with st.spinner("Ajustando GMM..."):
        detector = HMMRegimeDetector(n_states=n_states)
        try:
            detector.fit(X)
            states, probs = detector.predict_regime(X)
        except Exception as e:
            st.error(f"Modelo falhou: {e}"); return

    reg_names = [regime_map[s] for s in states]

    # ── Plot principal ────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=["Vendas EV + Regimes Detectados",
                         "Probabilidade por Regime (GMM)"],
    )

    # Linha de vendas
    fig.add_trace(go.Scatter(
        x=data["year"].values, y=data["ev_sales"].values / 1e3,
        mode="lines", name="Vendas EV (k)",
        line=dict(color=SLATE, width=1.5),
    ), row=1, col=1)

    # Pontos coloridos por regime
    for s in range(n_states):
        mask = np.array(states) == s
        if not mask.any(): continue
        fig.add_trace(go.Scatter(
            x=data["year"].values[mask],
            y=data["ev_sales"].values[mask] / 1e3,
            mode="markers", name=regime_map[s],
            marker=dict(size=9, color=color_map[s],
                        line=dict(width=1.2, color="white")),
        ), row=1, col=1)

    # Probabilidades empilhadas
    for s in range(probs.shape[1]):
        fig.add_trace(go.Scatter(
            x=data["year"].values, y=probs[:, s],
            mode="lines", name=regime_map[s],
            line=dict(color=color_map[s], width=0.8),
            fill="tonexty" if s > 0 else "tozeroy",
            fillcolor=color_map[s].replace("#", "rgba(").rstrip("") + "44)"
                      if False else f"rgba(0,0,0,0.04)",
            showlegend=False,
        ), row=2, col=1)

    # Layout
    lay = dict(
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(family="Inter, Arial, sans-serif", size=11, color=SLATE),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=BORDER,
                    borderwidth=1, font=dict(size=10)),
        yaxis=dict(title="k unidades", gridcolor=BORDER, gridwidth=0.5,
                   linecolor=BORDER, tickfont=dict(size=10)),
        yaxis2=dict(title="Probabilidade", range=[0, 1],
                    gridcolor=BORDER, gridwidth=0.5,
                    linecolor=BORDER, tickfont=dict(size=10)),
        xaxis2=dict(title="Ano", gridcolor=BORDER, linecolor=BORDER,
                    tickfont=dict(size=10)),
    )
    for ann in fig.layout.annotations:
        ann.font.color = SLATE
        ann.font.size  = 12
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True)

    # Tabela
    reg_df = pd.DataFrame({
        "Ano": data["year"].values,
        "Regime": reg_names,
        "Vendas (k)": (data["ev_sales"].values / 1e3).round(1),
        "YoY (%)": data["growth_rate"].fillna(0).round(1).values,
    })
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Bayesian Changepoint ──────────────────────────────────────────────
    st.subheader("📍 Bayesian Changepoint Detection")
    cp     = BayesianChangePoint()
    cp_pts = cp.detect(data["ev_sales"].values)

    if cp_pts:
        cp_years = [int(data["year"].values[i]) for i in cp_pts if i < len(data)]
        st.success(f"Pontos de mudança estrutural detectados: **{', '.join(map(str, cp_years))}**")

        fig_cp = go.Figure()
        fig_cp.add_trace(go.Scatter(
            x=data["year"].values, y=data["ev_sales"].values / 1e3,
            mode="lines+markers", name="Vendas EV",
            line=dict(color=NAVY, width=2),
            marker=dict(size=5, color=NAVY),
        ))
        for yr in cp_years:
            fig_cp.add_vline(
                x=yr, line_dash="dash", line_color=AMBER, line_width=1.5,
                annotation_text=str(yr),
                annotation_font_color=AMBER, annotation_font_size=10,
            )
        lay_cp = base_layout(height=280)
        lay_cp.update({"xaxis": {**lay_cp.get("xaxis", {}), "title": "Ano"},
                        "yaxis": {**lay_cp.get("yaxis", {}), "title": "k unidades"},
                        "showlegend": False})
        fig_cp.update_layout(**lay_cp)
        st.plotly_chart(fig_cp, use_container_width=True)
    else:
        st.info("Nenhum ponto de mudança estrutural detectado.")
