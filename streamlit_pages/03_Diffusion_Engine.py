# streamlit_pages/03_Diffusion_Engine.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel

_BG="#161b22"; _GRID="#21262d"; _TEXT="#e6edf3"
_COLORS = {"Bass":"#388bfd","Logistic":"#3fb950","Gompertz":"#d2a8ff"}


def render(df: pd.DataFrame):
    st.title("🧠 Diffusion Engine")
    st.markdown("Ajuste de modelos de difusão tecnológica (Bass, Logistic, Gompertz)")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    col_sel, col_mode = st.columns([2,1])
    with col_sel:
        selected = st.selectbox("País", countries,
                                index=countries.index("China") if "China" in countries else 0)
    with col_mode:
        mode = st.radio("Série", ["Adoção cumulativa", "Vendas anuais"], horizontal=True)

    data = df[df["country"] == selected].sort_values("year")
    if len(data) < 4:
        st.warning("Dados insuficientes para ajuste."); return

    t = np.arange(len(data), dtype=float)
    S = data["ev_sales"].cumsum().values  # cumulativo
    y_annual = data["ev_sales"].values

    # Horizonte de projeção
    horizon = st.slider("Horizonte de projeção (anos além do histórico)", 2, 20, 8)
    t_all   = np.arange(len(data) + horizon, dtype=float)
    years_all = np.arange(int(data["year"].min()), int(data["year"].min()) + len(t_all))

    models = {"Bass": BassDiffusion(), "Logistic": LogisticGrowth(), "Gompertz": GompertzModel()}
    fit_results = {}
    for name, model in models.items():
        try:
            model.fit(t, S)
            fit_results[name] = model
        except Exception as e:
            fit_results[name] = None

    # Métricas de ajuste
    st.markdown("#### Parâmetros estimados")
    cols = st.columns(3)
    for i, (name, model) in enumerate(fit_results.items()):
        with cols[i]:
            if model:
                p = model.get_params()
                st.success(f"**{name}**")
                for k,v in p.items():
                    st.metric(k, f"{v:.4f}")
            else:
                st.error(f"**{name}** — não convergiu")

    st.markdown("---")

    # Plot principal
    fig = go.Figure()
    if mode == "Adoção cumulativa":
        fig.add_trace(go.Scatter(
            x=data["year"].values, y=S/1e3,
            mode="markers+lines", name="Observado",
            marker=dict(size=7, color="#e6edf3"),
            line=dict(color="#e6edf3", width=1.5)
        ))
        for name, model in fit_results.items():
            if model:
                pred = model.predict(t_all)
                fig.add_trace(go.Scatter(
                    x=years_all, y=pred/1e3,
                    mode="lines", name=name,
                    line=dict(color=_COLORS[name], dash="dash", width=2)
                ))
        ylab = "Mil unidades (cumulativo)"
        title_plot = f"Curvas de Difusão — {selected} (Cumulativo)"
    else:
        fig.add_trace(go.Bar(
            x=data["year"].values, y=y_annual/1e3,
            name="Observado", marker_color="#8b949e"
        ))
        for name, model in fit_results.items():
            if model:
                pred_cum = model.predict(t_all)
                pred_ann = np.diff(pred_cum, prepend=pred_cum[0])
                fig.add_trace(go.Scatter(
                    x=years_all, y=pred_ann/1e3,
                    mode="lines", name=name,
                    line=dict(color=_COLORS[name], dash="dash", width=2)
                ))
        ylab = "Mil unidades (anual)"
        title_plot = f"Vendas Anuais Modeladas — {selected}"

    # Linha de separação histórico/projeção
    fig.add_vline(x=int(data["year"].max()), line_dash="dot",
                  line_color="#f0883e", annotation_text="Hoje",
                  annotation_font_color="#f0883e")
    fig.update_layout(
        title=title_plot, xaxis_title="Ano", yaxis_title=ylab,
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_TEXT), height=420,
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d"),
        xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID),
        margin=dict(l=0,r=0,t=40,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabela de projeções
    st.markdown("#### Projeções (anos futuros)")
    future_idx = np.arange(len(data), len(data)+horizon, dtype=float)
    future_years = np.arange(int(data["year"].max())+1, int(data["year"].max())+1+horizon)
    rows = []
    for yr, ti in zip(future_years, future_idx):
        row = {"Ano": yr}
        for name, model in fit_results.items():
            if model:
                row[f"{name} (k un.)"] = round(model.predict(np.array([ti]))[0]/1e3, 1)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
 