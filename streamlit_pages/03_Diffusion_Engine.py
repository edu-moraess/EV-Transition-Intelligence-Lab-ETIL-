# streamlit_pages/03_Diffusion_Engine.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel
from visualization.theme import (
    base_layout, NAVY, NAVY_LIGHT, RED_DARK,
    GREEN_MED, AMBER, SLATE, BORDER
)

_COLORS = {"Bass": NAVY, "Logistic": "#2e7d32", "Gompertz": AMBER}


def render(df: pd.DataFrame):
    st.title("🧠 Diffusion Engine")
    st.caption("Modelos Bass, Logistic e Gompertz — curvas de difusão tecnológica com projeções")
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
        st.warning("Dados insuficientes para ajuste (mín. 4 anos)."); return

    t      = np.arange(len(data), dtype=float)
    S      = data["ev_sales"].cumsum().values
    y_ann  = data["ev_sales"].values
    horizon = st.slider("Horizonte de projeção (anos)", 2, 20, 8)
    t_all  = np.arange(len(data) + horizon, dtype=float)
    yrs_all = np.arange(int(data["year"].min()),
                         int(data["year"].min()) + len(t_all))

    # ── Ajuste ────────────────────────────────────────────────────────────
    models  = {"Bass": BassDiffusion(), "Logistic": LogisticGrowth(), "Gompertz": GompertzModel()}
    fitted  = {}
    r2_vals = {}

    for name, model in models.items():
        try:
            model.fit(t, S)
            pred    = model.predict(t)
            ss_res  = np.sum((S - pred) ** 2)
            ss_tot  = np.sum((S - S.mean()) ** 2)
            r2_vals[name] = 1 - ss_res / ss_tot
            fitted[name]  = model
        except Exception:
            pass

    # ── Parâmetros ────────────────────────────────────────────────────────
    st.markdown("#### Parâmetros estimados")
    param_cols = st.columns(3)
    for i, (name, model) in enumerate(models.items()):
        with param_cols[i]:
            if name in fitted:
                r2 = r2_vals[name]
                st.success(f"**{name}** — R² = {r2:.4f}")
                for k, v in fitted[name].get_params().items():
                    st.metric(k, f"{v:.4f}")
            else:
                st.error(f"**{name}** — não convergiu")

    st.markdown("---")

    # ── Plot principal ────────────────────────────────────────────────────
    fig = go.Figure()

    if mode == "Cumulativo":
        fig.add_trace(go.Scatter(
            x=data["year"].values, y=S / 1e3,
            mode="markers+lines", name="Observado",
            line=dict(color=SLATE, width=1.5, dash="dot"),
            marker=dict(size=6, color=SLATE,
                        line=dict(width=1, color="white")),
        ))
        for name, model in fitted.items():
            fig.add_trace(go.Scatter(
                x=yrs_all, y=model.predict(t_all) / 1e3,
                mode="lines", name=f"{name} (R²={r2_vals[name]:.3f})",
                line=dict(color=_COLORS[name], width=2, dash="dash"),
            ))
        ylab  = "k unidades (cumulativo)"
        title = f"Difusão Cumulativa — {selected}"

    else:
        fig.add_trace(go.Bar(
            x=data["year"].values, y=y_ann / 1e3,
            name="Observado",
            marker_color=SLATE, opacity=0.45,
        ))
        for name, model in fitted.items():
            pred_cum = model.predict(t_all)
            pred_ann = np.diff(pred_cum, prepend=pred_cum[0])
            fig.add_trace(go.Scatter(
                x=yrs_all, y=pred_ann / 1e3,
                mode="lines", name=f"{name} (R²={r2_vals[name]:.3f})",
                line=dict(color=_COLORS[name], width=2, dash="dash"),
            ))
        ylab  = "k unidades (anual)"
        title = f"Vendas Anuais Modeladas — {selected}"

    fig.add_vline(
        x=int(data["year"].max()), line_dash="dot",
        line_color=SLATE, line_width=1,
        annotation_text="Hoje",
        annotation_font_color=SLATE, annotation_font_size=10,
    )

    lay = base_layout(height=420, title=title)
    lay.update({"xaxis": {**lay.get("xaxis", {}), "title": "Ano"},
                 "yaxis": {**lay.get("yaxis", {}), "title": ylab}})
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabela projeções ──────────────────────────────────────────────────
    if fitted:
        st.markdown("#### Projeções (anos futuros)")
        future_t    = np.arange(len(data), len(data) + horizon, dtype=float)
        future_yrs  = np.arange(int(data["year"].max()) + 1,
                                 int(data["year"].max()) + 1 + horizon)
        rows = []
        for yr, ti in zip(future_yrs, future_t):
            row = {"Ano": int(yr)}
            for name, model in fitted.items():
                val = model.predict(np.array([ti]))[0]
                row[f"{name} (k)"] = round(val / 1e3, 1)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Comparativo entre países ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Comparativo multi-país (Bass)")
    compare = st.multiselect(
        "Adicionar países",
        [c for c in countries if c != selected],
        default=["Norway", "Germany"][:2] if "Norway" in countries else [],
        max_selections=4,
    )
    if compare:
        fig_cmp = go.Figure()
        all_ctry = [selected] + compare
        palette  = [NAVY, "#2e7d32", AMBER, RED_DARK, NAVY_LIGHT]
        for ci, ctry in enumerate(all_ctry):
            d_c = df[df["country"] == ctry].sort_values("year")
            if len(d_c) < 4: continue
            t_c = np.arange(len(d_c), dtype=float)
            S_c = d_c["ev_sales"].cumsum().values
            fig_cmp.add_trace(go.Scatter(
                x=d_c["year"].values, y=S_c / 1e3,
                mode="markers", name=f"{ctry} (obs)",
                marker=dict(size=5, color=palette[ci % len(palette)]),
            ))
            try:
                bass = BassDiffusion()
                bass.fit(t_c, S_c)
                t_ext = np.arange(0, len(t_c) + horizon, dtype=float)
                y_ext = np.arange(int(d_c["year"].min()),
                                   int(d_c["year"].min()) + len(t_ext))
                fig_cmp.add_trace(go.Scatter(
                    x=y_ext, y=bass.predict(t_ext) / 1e3,
                    mode="lines", name=f"{ctry} (Bass)",
                    line=dict(color=palette[ci % len(palette)], width=1.8, dash="dash"),
                ))
            except Exception:
                pass

        fig_cmp.add_vline(x=int(df["year"].max()), line_dash="dot",
                           line_color=SLATE, line_width=1)
        lay_c = base_layout(height=360)
        lay_c.update({"xaxis": {**lay_c.get("xaxis", {}), "title": "Ano"},
                       "yaxis": {**lay_c.get("yaxis", {}),
                                  "title": "k unidades (cumulativo)"}})
        fig_cmp.update_layout(**lay_c)
        st.plotly_chart(fig_cmp, use_container_width=True)
