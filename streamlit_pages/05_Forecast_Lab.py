# streamlit_pages/05_Forecast_Lab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models.forecasting import BayesianGPForecast, MonteCarloSimulator
from visualization.theme import chart_config, base_layout, NAVY, RED_DARK, GREEN_MED, AMBER, SLATE

_SCENARIO_COLOR  = {"base": NAVY, "optimistic": GREEN_MED, "pessimistic": RED_DARK}
_SCENARIO_FACTOR = {"base": 1.0,  "optimistic": 1.25,      "pessimistic": 0.75}


def render(df: pd.DataFrame):
    st.markdown("## Forecast Lab")
    st.markdown(
        "A single point forecast can hide uncertainty. This page uses a **Bayesian Gaussian Process** to model "
        "the trend with uncertainty, then runs **2,000 Monte Carlo simulations** to produce "
        "an empirical distribution of possible futures. "
        "The result is a scenario-conditioned range of simulated outcomes rather than a single estimate."
    )
    st.markdown("---")

    countries = sorted(df["country"].unique())
    col_s, col_h, col_sc = st.columns([2, 1, 1])
    with col_s:
        selected = st.selectbox("Country", countries,
                                index=countries.index("Germany") if "Germany" in countries else 0)
    with col_h:
        horizon = st.slider("Horizon (years)", 3, 20, 10)
    with col_sc:
        scenario = st.selectbox("Scenario", ["base", "optimistic", "pessimistic"])

    data    = df[df["country"] == selected].sort_values("year").reset_index(drop=True)
    if len(data) < 4:
        st.warning("Insufficient data (minimum 4 years)."); return

    t_h    = np.arange(len(data), dtype=float).reshape(-1, 1)
    y_h    = data["ev_sales"].values.astype(float)
    t_f    = np.arange(len(data), len(data) + horizon, dtype=float).reshape(-1, 1)
    yrs_h  = data["year"].values
    yrs_f  = np.arange(int(data["year"].max()) + 1,
                        int(data["year"].max()) + 1 + horizon)
    factor = _SCENARIO_FACTOR[scenario]
    color  = _SCENARIO_COLOR[scenario]

    with st.spinner("Fitting Gaussian Process..."):
        gp = BayesianGPForecast()
        try:
            gp.fit(t_h, y_h)
        except Exception as e:
            st.error(f"GP fit failed: {e}"); return

    mc   = MonteCarloSimulator(n_simulations=2000)
    sims = np.maximum(mc.simulate_from_gp(gp, t_f), 0) * factor
    lo, med, hi = mc.get_percentiles(sims)

    last_hist   = y_h[-1]
    proj_median = float(med[-1])
    cagr_proj   = ((proj_median / last_hist) ** (1 / horizon) - 1) * 100 if last_hist > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Median projection — {yrs_f[-1]}",  f"{proj_median/1e3:,.1f}k units")
    c2.metric("Projected CAGR (median)",           f"{cagr_proj:.1f}%" if not np.isnan(cagr_proj) else "N/A")
    c3.metric("P10 / P90 — terminal year",
              f"{float(np.percentile(sims[:,-1],10))/1e3:.0f}k / "
              f"{float(np.percentile(sims[:,-1],90))/1e3:.0f}k")
    c4.metric("Active scenario",          scenario.capitalize())

    st.markdown("---")

    # ── Forecast chart ────────────────────────────────────────────────────
    fig = go.Figure()
    lo80 = np.percentile(sims, 10, axis=0)
    hi80 = np.percentile(sims, 90, axis=0)

    fig.add_trace(go.Scatter(
        x=np.concatenate([yrs_f, yrs_f[::-1]]),
        y=np.concatenate([hi / 1e3, lo[::-1] / 1e3]),
        fill="toself", fillcolor="rgba(31,58,110,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% empirical interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([yrs_f, yrs_f[::-1]]),
        y=np.concatenate([hi80 / 1e3, lo80[::-1] / 1e3]),
        fill="toself", fillcolor="rgba(31,58,110,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% empirical interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=yrs_f, y=med / 1e3, mode="lines",
        name=f"Median ({scenario})",
        line=dict(color=color, width=2.5, dash="dash"),
        hovertemplate="%{x}: %{y:.1f}k<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=yrs_h, y=y_h / 1e3, mode="lines+markers",
        name="Historical",
        line=dict(color=SLATE, width=2),
        marker=dict(size=5, color=SLATE, line=dict(width=1, color="white")),
        hovertemplate="%{x}: %{y:.1f}k<extra></extra>",
    ))
    fig.add_vline(x=int(data["year"].max()), line_dash="dot",
                  line_color=SLATE, line_width=1,
                  annotation_text="Last observed year",
                  annotation_font_color=SLATE, annotation_font_size=10)

    lay = base_layout(420, f"Gaussian Process + Monte Carlo Forecast — {selected} ({scenario})")
    lay.update({"xaxis": {**lay.get("xaxis", {}), "title": "Year"},
                 "yaxis": {**lay.get("yaxis", {}), "title": "Thousand units"}})
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True, config=chart_config())

    # ── Terminal distribution ──────────────────────────────────────────────
    st.markdown(f"#### Terminal-year distribution — {yrs_f[-1]}")
    final = sims[:, -1] / 1e3
    p5, p25, p50, p75, p95 = np.percentile(final, [5, 25, 50, 75, 95])

    col_hist, col_pct = st.columns([2, 1])
    with col_hist:
        fig_h = go.Figure(go.Histogram(
            x=final, nbinsx=60, marker_color=NAVY, opacity=0.75,
        ))
        for pct, lbl, clr in [(5,"P5",RED_DARK),(50,"P50",GREEN_MED),(95,"P95",AMBER)]:
            fig_h.add_vline(x=float(np.percentile(final, pct)),
                             line_dash="dash", line_color=clr, line_width=1.5,
                             annotation_text=lbl,
                             annotation_font_color=clr, annotation_font_size=10)
        lay_h = base_layout(240)
        lay_h.update({"xaxis": {**lay_h.get("xaxis", {}), "title": "Thousand units"},
                       "yaxis": {**lay_h.get("yaxis", {}), "title": "Frequency"},
                       "showlegend": False})
        fig_h.update_layout(**lay_h)
        st.plotly_chart(fig_h, use_container_width=True, config=chart_config())

    with col_pct:
        st.markdown("**Percentiles**")
        for lbl, val in [("P5", p5),("P25", p25),("P50", p50),("P75", p75),("P95", p95)]:
            st.metric(lbl, f"{val:.1f}k")

    # ── Percentile table by year ───────────────────────────────────────────
    st.markdown("#### Percentiles by Year (thousand units)")
    pct_rows = [{
        "Year": int(yr),
        "P10":  round(float(np.percentile(sims[:, i] / 1e3, 10)), 1),
        "P25":  round(float(np.percentile(sims[:, i] / 1e3, 25)), 1),
        "P50":  round(float(np.percentile(sims[:, i] / 1e3, 50)), 1),
        "P75":  round(float(np.percentile(sims[:, i] / 1e3, 75)), 1),
        "P90":  round(float(np.percentile(sims[:, i] / 1e3, 90)), 1),
    } for i, yr in enumerate(yrs_f)]
    st.dataframe(pd.DataFrame(pct_rows), use_container_width=True, hide_index=True)
