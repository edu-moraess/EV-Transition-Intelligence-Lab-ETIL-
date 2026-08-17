# streamlit_pages/04_Regime_Detection.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models.regimes import HMMRegimeDetector, BayesianChangePoint
from visualization.theme import chart_config, base_layout, NAVY, NAVY_LIGHT, RED_DARK, GREEN_MED, AMBER, SLATE, BORDER

_REGIME_LABELS = {0: "Emerging", 1: "Expansion", 2: "Acceleration", 3: "Maturity", 4: "Peak"}
_REGIME_COLORS = {0: SLATE, 1: NAVY_LIGHT, 2: GREEN_MED, 3: AMBER, 4: RED_DARK}


def render(df: pd.DataFrame):
    st.markdown("## Regime Detection")
    st.markdown(
        "Not all EV markets are in the same phase. A **Gaussian Mixture Model** assigns each "
        "year in a country's history to a configurable number of ordered adoption regimes — Emerging, Expansion, "
        "Acceleration, Maturity or Peak. A **mean-shift screening rule** highlights candidate structural breaks; "
        "it does not establish causality or identify an economic event by itself."
    )
    st.markdown("---")

    countries = sorted(df["country"].unique())
    col_s, col_n = st.columns([3, 1])
    with col_s:
        selected = st.selectbox("Country", countries,
                                index=countries.index("Norway") if "Norway" in countries else 0)
    with col_n:
        n_states = st.selectbox("Number of regimes", [3, 4, 5], index=1)

    data = df[df["country"] == selected].sort_values("year").reset_index(drop=True)
    if len(data) < 5:
        st.warning("Insufficient data (minimum 5 years required)."); return

    feat_cols = []
    if "ev_stock_share" in data.columns and data["ev_stock_share"].notna().any():
        feat_cols.append("ev_stock_share")
    elif "penetration" in data.columns and data["penetration"].notna().any():
        feat_cols.append("penetration")
    if "growth_rate" in data.columns and data["growth_rate"].notna().any():
        feat_cols.append("growth_rate")

    if not feat_cols:
        st.error("Insufficient features for regime detection."); return

    X = np.column_stack([data[c].fillna(0).values for c in feat_cols])
    regime_map = {i: _REGIME_LABELS.get(i, f"R{i}") for i in range(n_states)}
    color_map  = {i: _REGIME_COLORS.get(i, SLATE) for i in range(n_states)}

    with st.spinner("Fitting GMM..."):
        detector = HMMRegimeDetector(n_states=n_states)
        try:
            detector.fit(X)
            states, probs = detector.predict_regime(X)
        except Exception as e:
            st.error(f"Model failed: {e}"); return

    reg_names = [regime_map[s] for s in states]

    # ── Main chart ────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.06,
        subplot_titles=["EV Sales and Model-Assigned Regimes",
                         "Regime Probability (GMM)"],
    )
    # FIX: usar NAVY para linha de vendas — mais visível e distinto da legenda SLATE
    fig.add_trace(go.Scatter(
        x=data["year"].values, y=data["ev_sales"].values / 1e3,
        mode="lines", name="EV Sales (k)",
        line=dict(color=NAVY, width=1.5),
    ), row=1, col=1)

    for s in range(n_states):
        mask = np.array(states) == s
        if not mask.any(): continue
        fig.add_trace(go.Scatter(
            x=data["year"].values[mask], y=data["ev_sales"].values[mask] / 1e3,
            mode="markers", name=regime_map[s],
            marker=dict(size=9, color=color_map[s],
                        line=dict(width=1.2, color="white")),
        ), row=1, col=1)

    # FIX: remover fill acumulativo que criava retângulo cinza
    for s in range(probs.shape[1]):
        fig.add_trace(go.Scatter(
            x=data["year"].values, y=probs[:, s],
            mode="lines", name=regime_map[s],
            line=dict(color=color_map[s], width=1.2),
            # fill removido — linhas puras sem artefato visual
            showlegend=False,
        ), row=2, col=1)

    # Aplicar base_layout para consistência de tema
    lay = base_layout(520)
    lay.update({
        "margin": dict(l=0, r=0, t=40, b=0),
        "legend": dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(128,128,128,0.25)", borderwidth=1,
            font=dict(size=10),
        ),
        "yaxis": dict(
            title="Thousand units",
            gridcolor="rgba(128,128,128,0.15)", gridwidth=0.5,
            linecolor="rgba(128,128,128,0.2)", tickfont=dict(size=10),
        ),
        "yaxis2": dict(
            title="Probability", range=[0, 1],
            gridcolor="rgba(128,128,128,0.15)", gridwidth=0.5,
            linecolor="rgba(128,128,128,0.2)", tickfont=dict(size=10),
        ),
        "xaxis2": dict(
            title="Year", gridcolor="rgba(128,128,128,0.15)",
            linecolor="rgba(128,128,128,0.2)", tickfont=dict(size=10),
        ),
    })
    for ann in fig.layout.annotations:
        ann.font.size = 12
    fig.update_layout(**lay)
    st.plotly_chart(fig, config=chart_config())

    st.dataframe(pd.DataFrame({
        "Year":       data["year"].values,
        "Regime":     reg_names,
        "EV Sales (k)": (data["ev_sales"].values / 1e3).round(1),
        "YoY (%)":    data["growth_rate"].fillna(0).round(1).values,
    }), hide_index=True)

    st.markdown("---")
    st.subheader("Candidate Structural-Break Signals")
    cp     = BayesianChangePoint()
    cp_pts = cp.detect(data["ev_sales"].values)

    if cp_pts:
        cp_years = [int(data["year"].values[i]) for i in cp_pts if i < len(data)]
        st.info(f"Candidate mean-shift signals: **{', '.join(map(str, cp_years))}**")
        fig_cp = go.Figure(go.Scatter(
            x=data["year"].values, y=data["ev_sales"].values / 1e3,
            mode="lines+markers", line=dict(color=NAVY, width=2),
            marker=dict(size=5, color=NAVY),
        ))
        for yr in cp_years:
            fig_cp.add_vline(x=yr, line_dash="dash", line_color=AMBER, line_width=1.5,
                              annotation_text=str(yr),
                              annotation_font_color=AMBER, annotation_font_size=10)
        lay_cp = base_layout(280)
        lay_cp.update({"xaxis": {**lay_cp.get("xaxis", {}), "title": "Year"},
                        "yaxis": {**lay_cp.get("yaxis", {}), "title": "Thousand units"},
                        "showlegend": False})
        fig_cp.update_layout(**lay_cp)
        st.plotly_chart(fig_cp, config=chart_config())
    else:
        st.info("No candidate mean-shift signals detected.")
