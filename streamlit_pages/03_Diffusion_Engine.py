# streamlit_pages/03_Diffusion_Engine.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel

_COLORS = {"Bass": "#636EFA", "Logistic": "#00CC96", "Gompertz": "#AB63FA"}


def render(df: pd.DataFrame):
    st.title("🧠 Diffusion Engine")
    st.caption("Modelos Bass, Logistic e Gompertz — curvas de difusão tecnológica")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    col_s, col_m = st.columns([2, 1])
    with col_s:
        selected = st.selectbox("País", countries,
                                index=countries.index("China") if "China" in countries else 0)
    with col_m:
        mode = st.radio("Série", ["Cumulativo", "Anual"], horizontal=True)

    data = df[df["country"] == selected].sort_values("year")
    if len(data) < 4:
        st.warning("Dados insuficientes para ajuste."); return

    t      = np.arange(len(data), dtype=float)
    S      = data["ev_sales"].cumsum().values
    y_ann  = data["ev_sales"].values
    horizon = st.slider("Horizonte de projeção (anos)", 2, 20, 8)
    t_all  = np.arange(len(data) + horizon, dtype=float)
    yrs_all = np.arange(int(data["year"].min()), int(data["year"].min()) + len(t_all))

    models = {"Bass": BassDiffusion(), "Logistic": LogisticGrowth(), "Gompertz": GompertzModel()}
    fitted = {}
    param_cols = st.columns(3)
    for i, (name, model) in enumerate(models.items()):
        try:
            model.fit(t, S)
            fitted[name] = model
            with param_cols[i]:
                st.success(f"**{name}**")
                for k, v in model.get_params().items():
                    st.metric(k, f"{v:.4f}")
        except Exception:
            with param_cols[i]:
                st.error(f"**{name}** não convergiu")

    st.markdown("---")

    fig = go.Figure()
    if mode == "Cumulativo":
        fig.add_trace(go.Scatter(x=data["year"].values, y=S/1e3,
                                  mode="markers+lines", name="Observado",
                                  line=dict(width=2)))
        for name, model in fitted.items():
            fig.add_trace(go.Scatter(x=yrs_all, y=model.predict(t_all)/1e3,
                                      mode="lines", name=name,
                                      line=dict(color=_COLORS[name], dash="dash", width=2)))
        ylab, title = "k unidades (cumulativo)", f"Difusão Cumulativa — {selected}"
    else:
        fig.add_trace(go.Bar(x=data["year"].values, y=y_ann/1e3, name="Observado", opacity=0.6))
        for name, model in fitted.items():
            pred_cum = model.predict(t_all)
            pred_ann = np.diff(pred_cum, prepend=pred_cum[0])
            fig.add_trace(go.Scatter(x=yrs_all, y=pred_ann/1e3, mode="lines", name=name,
                                      line=dict(color=_COLORS[name], dash="dash", width=2)))
        ylab, title = "k unidades (anual)", f"Vendas Anuais Modeladas — {selected}"

    fig.add_vline(x=int(data["year"].max()), line_dash="dot", annotation_text="Hoje")
    fig.update_layout(xaxis_title="Ano", yaxis_title=ylab, title=title,
                       height=420, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Tabela projeções
    future_t = np.arange(len(data), len(data)+horizon, dtype=float)
    future_y = np.arange(int(data["year"].max())+1, int(data["year"].max())+1+horizon)
    rows = []
    for yr, ti in zip(future_y, future_t):
        row = {"Ano": yr}
        for name, model in fitted.items():
            row[f"{name} (k)"] = round(model.predict(np.array([ti]))[0]/1e3, 1)
        rows.append(row)
    if rows:
        st.markdown("#### Projeções (anos futuros)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
