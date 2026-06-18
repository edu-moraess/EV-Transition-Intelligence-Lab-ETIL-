# streamlit_pages/05_Forecast_Lab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models.forecasting import BayesianGPForecast, MonteCarloSimulator
from visualization.theme import (
    base_layout, NAVY, NAVY_LIGHT, NAVY_PALE,
    RED_DARK, GREEN_MED, AMBER, SLATE, BORDER
)

_SCENARIO_COLOR = {"base": NAVY, "optimistic": GREEN_MED, "pessimistic": RED_DARK}
_SCENARIO_FACTOR = {"base": 1.0, "optimistic": 1.25, "pessimistic": 0.75}


def render(df: pd.DataFrame):
    st.title("📊 Forecast Lab")
    st.caption("Gaussian Process Regression + Simulação Monte Carlo com análise de cenários")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    col_s, col_h, col_sc = st.columns([2, 1, 1])
    with col_s:
        selected = st.selectbox("País", countries,
                                index=countries.index("Germany") if "Germany" in countries else 0)
    with col_h:
        horizon = st.slider("Horizonte (anos)", 3, 20, 10)
    with col_sc:
        scenario = st.selectbox("Cenário", ["base", "optimistic", "pessimistic"])

    data    = df[df["country"] == selected].sort_values("year").reset_index(drop=True)
    if len(data) < 4:
        st.warning("Dados insuficientes (mín. 4 anos)."); return

    t_h   = np.arange(len(data), dtype=float).reshape(-1, 1)
    y_h   = data["ev_sales"].values.astype(float)
    t_f   = np.arange(len(data), len(data) + horizon, dtype=float).reshape(-1, 1)
    yrs_h = data["year"].values
    yrs_f = np.arange(int(data["year"].max()) + 1,
                       int(data["year"].max()) + 1 + horizon)
    factor = _SCENARIO_FACTOR[scenario]
    color  = _SCENARIO_COLOR[scenario]

    with st.spinner("Ajustando Gaussian Process..."):
        gp = BayesianGPForecast()
        try:
            gp.fit(t_h, y_h)
        except Exception as e:
            st.error(f"GP falhou: {e}"); return

    mean, std = gp.predict(t_f)
    mean = np.maximum(mean, 0)

    mc   = MonteCarloSimulator(n_simulations=2000)
    sims = np.maximum(mc.simulate_from_gp(gp, t_f), 0) * factor
    lo, med, hi = mc.get_percentiles(sims)

    # KPIs
    last_hist   = y_h[-1]
    proj_median = float(med[-1])
    cagr_proj   = ((proj_median / last_hist) ** (1 / horizon) - 1) * 100 if last_hist > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Projeção {yrs_f[-1]}",    f"{proj_median/1e3:,.1f}k un.")
    c2.metric("CAGR projetado",            f"{cagr_proj:.1f}%" if not np.isnan(cagr_proj) else "N/A")
    c3.metric("P10 / P90",                f"{float(np.percentile(sims[:,-1],10))/1e3:.0f}k / "
                                           f"{float(np.percentile(sims[:,-1],90))/1e3:.0f}k")
    c4.metric("Cenário",                   scenario.capitalize())

    st.markdown("---")

    # ── Forecast plot ──────────────────────────────────────────────────────
    fig = go.Figure()

    # CI 80%
    lo80 = np.percentile(sims, 10, axis=0)
    hi80 = np.percentile(sims, 90, axis=0)
    fig.add_trace(go.Scatter(
        x=np.concatenate([yrs_f, yrs_f[::-1]]),
        y=np.concatenate([hi80 / 1e3, lo80[::-1] / 1e3]),
        fill="toself",
        fillcolor=f"rgba(31,58,110,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IC 80%", showlegend=True,
        hoverinfo="skip",
    ))

    # CI 95%
    fig.add_trace(go.Scatter(
        x=np.concatenate([yrs_f, yrs_f[::-1]]),
        y=np.concatenate([hi / 1e3, lo[::-1] / 1e3]),
        fill="toself",
        fillcolor=f"rgba(31,58,110,0.05)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IC 95%", showlegend=True,
        hoverinfo="skip",
    ))

    # Mediana projetada
    fig.add_trace(go.Scatter(
        x=yrs_f, y=med / 1e3,
        mode="lines", name=f"Mediana ({scenario})",
        line=dict(color=color, width=2.5, dash="dash"),
        hovertemplate="%{x}: %{y:.1f}k<extra></extra>",
    ))

    # Histórico
    fig.add_trace(go.Scatter(
        x=yrs_h, y=y_h / 1e3,
        mode="lines+markers", name="Histórico",
        line=dict(color=SLATE, width=2),
        marker=dict(size=5, color=SLATE, line=dict(width=1, color="white")),
        hovertemplate="%{x}: %{y:.1f}k<extra></extra>",
    ))

    fig.add_vline(x=int(data["year"].max()), line_dash="dot",
                  line_color=SLATE, line_width=1,
                  annotation_text="Hoje",
                  annotation_font_color=SLATE, annotation_font_size=10)

    lay = base_layout(height=420, title=f"Forecast GP + MC — {selected} ({scenario})")
    lay.update({"xaxis": {**lay.get("xaxis", {}), "title": "Ano"},
                 "yaxis": {**lay.get("yaxis", {}), "title": "k unidades"}})
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True)

    # ── Distribuição MC ────────────────────────────────────────────────────
    st.markdown(f"#### Distribuição Monte Carlo — {yrs_f[-1]}")
    final = sims[:, -1] / 1e3
    p5, p25, p50, p75, p95 = np.percentile(final, [5, 25, 50, 75, 95])

    # Box plot + histograma lado a lado
    col_hist, col_box = st.columns([2, 1])
    with col_hist:
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=final, nbinsx=60,
            marker_color=NAVY, opacity=0.75, name="Simulações",
        ))
        for pct, lbl, clr in [(5,"P5",RED_DARK),(50,"P50",GREEN_MED),(95,"P95",AMBER)]:
            fig_h.add_vline(x=float(np.percentile(final, pct)),
                             line_dash="dash", line_color=clr, line_width=1.5,
                             annotation_text=lbl,
                             annotation_font_color=clr, annotation_font_size=10)
        lay_h = base_layout(height=240)
        lay_h.update({"xaxis": {**lay_h.get("xaxis", {}), "title": "k unidades"},
                       "yaxis": {**lay_h.get("yaxis", {}), "title": "Frequência"},
                       "showlegend": False})
        fig_h.update_layout(**lay_h)
        st.plotly_chart(fig_h, use_container_width=True)

    with col_box:
        st.markdown("**Percentis finais**")
        st.metric("P5",  f"{p5:.1f}k")
        st.metric("P25", f"{p25:.1f}k")
        st.metric("P50", f"{p50:.1f}k")
        st.metric("P75", f"{p75:.1f}k")
        st.metric("P95", f"{p95:.1f}k")

    # ── Tabela percentis por ano ───────────────────────────────────────────
    st.markdown("#### Percentis por ano (k unidades)")
    pct_rows = [{
        "Ano": int(yr),
        "P10": round(float(np.percentile(sims[:, i] / 1e3, 10)), 1),
        "P25": round(float(np.percentile(sims[:, i] / 1e3, 25)), 1),
        "P50": round(float(np.percentile(sims[:, i] / 1e3, 50)), 1),
        "P75": round(float(np.percentile(sims[:, i] / 1e3, 75)), 1),
        "P90": round(float(np.percentile(sims[:, i] / 1e3, 90)), 1),
    } for i, yr in enumerate(yrs_f)]
    st.dataframe(pd.DataFrame(pct_rows), use_container_width=True, hide_index=True)
