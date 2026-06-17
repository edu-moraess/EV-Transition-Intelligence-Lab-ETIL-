# streamlit_pages/05_Forecast_Lab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models.forecasting import BayesianGPForecast, MonteCarloSimulator

_BG="#161b22"; _GRID="#21262d"; _TEXT="#e6edf3"


def render(df: pd.DataFrame):
    st.title("📊 Forecast Lab")
    st.markdown("Previsão Bayesiana via Gaussian Process + Simulação Monte Carlo")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    col_s, col_h, col_sc = st.columns([2,1,1])
    with col_s:
        selected = st.selectbox("País", countries,
                                index=countries.index("Germany") if "Germany" in countries else 0)
    with col_h:
        horizon = st.slider("Horizonte (anos)", 3, 20, 10)
    with col_sc:
        scenario = st.selectbox("Cenário", ["base","optimistic","pessimistic"])

    data = df[df["country"] == selected].sort_values("year").reset_index(drop=True)
    if len(data) < 4:
        st.warning("Dados insuficientes."); return

    t_hist   = np.arange(len(data), dtype=float).reshape(-1,1)
    y_hist   = data["ev_sales"].values.astype(float)
    t_future = np.arange(len(data), len(data)+horizon, dtype=float).reshape(-1,1)
    years_hist   = data["year"].values
    years_future = np.arange(int(data["year"].max())+1, int(data["year"].max())+1+horizon)

    with st.spinner("Ajustando Gaussian Process..."):
        gp = BayesianGPForecast()
        try:
            gp.fit(t_hist, y_hist)
        except Exception as e:
            st.error(f"GP falhou: {e}"); return

    mean, std = gp.predict(t_future)
    mean = np.maximum(mean, 0)

    # KPIs do forecast
    factor = {"base":1.0,"optimistic":1.2,"pessimistic":0.8}[scenario]
    med_last = mean[-1] * factor
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Projeção {years_future[-1]}", f"{med_last/1e3:,.1f}k unidades")
    c2.metric("CAGR projetado", f"{((med_last/y_hist[-1])**(1/horizon)-1)*100:.1f}%")
    c3.metric("Cenário ativo", scenario.capitalize())

    st.markdown("---")

    # Monte Carlo
    with st.spinner("Simulação Monte Carlo..."):
        mc = MonteCarloSimulator(n_simulations=2000)
        sims = mc.simulate_from_gp(gp, t_future)
        sims = np.maximum(sims, 0)
        adj_sims = mc.scenario_adjust(sims, scenario)
        lower, median, upper = mc.get_percentiles(adj_sims)

    # Plot
    fig = go.Figure()

    # Intervalo de confiança
    fig.add_trace(go.Scatter(
        x=np.concatenate([years_future, years_future[::-1]]),
        y=np.concatenate([upper/1e3, lower[::-1]/1e3]),
        fill="toself", fillcolor="rgba(56,139,253,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI", showlegend=True
    ))

    # Mediana projetada
    fig.add_trace(go.Scatter(
        x=years_future, y=median/1e3,
        mode="lines", name=f"Mediana ({scenario})",
        line=dict(color="#388bfd", width=2.5, dash="dash")
    ))

    # Histórico
    fig.add_trace(go.Scatter(
        x=years_hist, y=y_hist/1e3,
        mode="lines+markers", name="Histórico",
        line=dict(color="#e6edf3", width=2),
        marker=dict(size=5, color="#e6edf3")
    ))

    # Linha de separação
    fig.add_vline(x=int(data["year"].max()), line_dash="dot",
                  line_color="#f0883e", annotation_text="Hoje",
                  annotation_font_color="#f0883e")

    fig.update_layout(
        title=f"Forecast GP + MC — {selected} (cenário: {scenario})",
        xaxis_title="Ano", yaxis_title="Mil unidades",
        paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_TEXT),
        height=440, margin=dict(l=0,r=0,t=40,b=0),
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d"),
        xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribuição no horizonte final
    st.markdown(f"#### Distribuição Monte Carlo — {years_future[-1]}")
    final_vals = adj_sims[:, -1] / 1e3
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=final_vals, nbinsx=60,
        marker_color="#388bfd", opacity=0.8, name="Simulações"
    ))
    fig_hist.add_vline(x=float(np.percentile(final_vals,5)),  line_dash="dash", line_color="#f85149", annotation_text="P5")
    fig_hist.add_vline(x=float(np.percentile(final_vals,50)), line_dash="dash", line_color="#3fb950", annotation_text="P50")
    fig_hist.add_vline(x=float(np.percentile(final_vals,95)), line_dash="dash", line_color="#f0883e", annotation_text="P95")
    fig_hist.update_layout(
        xaxis_title="Mil unidades", yaxis_title="Frequência",
        paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_TEXT),
        height=280, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID),
        showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Tabela de percentis
    st.markdown("#### Percentis por ano (mil unidades)")
    pct_rows = []
    for i, yr in enumerate(years_future):
        col_vals = adj_sims[:,i]/1e3
        pct_rows.append({
            "Ano": int(yr),
            "P10": round(float(np.percentile(col_vals,10)),1),
            "P25": round(float(np.percentile(col_vals,25)),1),
            "P50 (mediana)": round(float(np.percentile(col_vals,50)),1),
            "P75": round(float(np.percentile(col_vals,75)),1),
            "P90": round(float(np.percentile(col_vals,90)),1),
        })
    st.dataframe(pd.DataFrame(pct_rows), use_container_width=True, hide_index=True)
