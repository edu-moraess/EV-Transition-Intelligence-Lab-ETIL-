# streamlit_pages/05_Forecast_Lab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from models.forecasting import BayesianGPForecast, MonteCarloSimulator


def render(df: pd.DataFrame):
    st.title("📊 Forecast Lab")
    st.caption("Gaussian Process Regression + Simulação Monte Carlo")
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

    data     = df[df["country"] == selected].sort_values("year").reset_index(drop=True)
    if len(data) < 4:
        st.warning("Dados insuficientes."); return

    t_h  = np.arange(len(data), dtype=float).reshape(-1, 1)
    y_h  = data["ev_sales"].values.astype(float)
    t_f  = np.arange(len(data), len(data)+horizon, dtype=float).reshape(-1, 1)
    yrs_h = data["year"].values
    yrs_f = np.arange(int(data["year"].max())+1, int(data["year"].max())+1+horizon)
    factor = {"base": 1.0, "optimistic": 1.2, "pessimistic": 0.8}[scenario]

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
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Projeção {yrs_f[-1]}", f"{med[-1]/1e3:,.1f}k un.")
    c2.metric("CAGR projetado", f"{((med[-1]/y_h[-1])**(1/horizon)-1)*100:.1f}%")
    c3.metric("Cenário", scenario.capitalize())

    st.markdown("---")

    # Plot forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([yrs_f, yrs_f[::-1]]),
        y=np.concatenate([hi/1e3, lo[::-1]/1e3]),
        fill="toself", fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"
    ))
    fig.add_trace(go.Scatter(
        x=yrs_f, y=med/1e3, mode="lines", name=f"Mediana ({scenario})",
        line=dict(width=2.5, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=yrs_h, y=y_h/1e3, mode="lines+markers", name="Histórico",
        line=dict(width=2), marker=dict(size=5)
    ))
    fig.add_vline(x=int(data["year"].max()), line_dash="dot", annotation_text="Hoje")
    fig.update_layout(
        title=f"Forecast GP + MC — {selected} ({scenario})",
        xaxis_title="Ano", yaxis_title="k unidades",
        height=440, margin=dict(l=0,r=0,t=40,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribuição final
    st.markdown(f"#### Distribuição Monte Carlo — {yrs_f[-1]}")
    final = sims[:, -1] / 1e3
    fig_h = px.histogram(x=final, nbins=60, labels={"x": "k unidades"},
                          color_discrete_sequence=["#636EFA"], opacity=0.8)
    for pct, label, color in [(5, "P5", "red"), (50, "P50", "green"), (95, "P95", "orange")]:
        fig_h.add_vline(x=float(np.percentile(final, pct)), line_dash="dash",
                         line_color=color, annotation_text=label)
    fig_h.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
    st.plotly_chart(fig_h, use_container_width=True)

    # Percentis por ano
    pct_rows = [{"Ano": int(yr), "P10": round(float(np.percentile(sims[:,i]/1e3,10)),1),
                  "P25": round(float(np.percentile(sims[:,i]/1e3,25)),1),
                  "P50": round(float(np.percentile(sims[:,i]/1e3,50)),1),
                  "P75": round(float(np.percentile(sims[:,i]/1e3,75)),1),
                  "P90": round(float(np.percentile(sims[:,i]/1e3,90)),1)}
                 for i, yr in enumerate(yrs_f)]
    st.markdown("#### Percentis por ano (k unidades)")
    st.dataframe(pd.DataFrame(pct_rows), use_container_width=True, hide_index=True)
