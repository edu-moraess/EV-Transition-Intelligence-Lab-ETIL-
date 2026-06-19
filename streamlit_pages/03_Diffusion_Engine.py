# streamlit_pages/03_Diffusion_Engine.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel
from visualization.theme import base_layout, NAVY, AMBER, SLATE

_COLORS = {"Bass": NAVY, "Logistic": "#2e7d32", "Gompertz": AMBER}


def render(df: pd.DataFrame):
    # ── CSS para modebar transparente ──────────────────────────────────────
    st.markdown("""
    <style>
    .modebar {
        background-color: transparent !important;
        box-shadow: none !important;
        border-radius: 8px !important;
        padding: 4px 8px !important;
    }
    .modebar .icon {
        fill: #1f3a6e !important;
        opacity: 0.7 !important;
        transition: opacity 0.2s !important;
    }
    .modebar .icon:hover {
        fill: #0d1117 !important;
        opacity: 1 !important;
    }
    .modebar-btn:hover {
        background-color: rgba(31, 58, 110, 0.08) !important;
        border-radius: 4px !important;
    }
    .modebar-group {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Configuração padrão para os gráficos ──────────────────────────────
    PLOTLY_CONFIG = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'zoom2d', 'pan2d', 'select2d', 'lasso2d',
            'zoomIn2d', 'zoomOut2d', 'autoScale2d',
            'resetScale2d', 'toImage', 'sendDataToCloud'
        ],
        'responsive': True,
    }

    st.markdown("## Diffusion Engine")
    st.markdown(
        "Technology adoption follows S-shaped curves — slow at first, then explosive, then plateauing. "
        "Three models capture this: **Bass** models innovation vs imitation; "
        "**Logistic** fits a symmetric S-curve; **Gompertz** allows for asymmetric, slower early growth. "
        "Fit them to any country and project where the market is heading."
    )
    st.markdown("---")

    countries = sorted(df["country"].unique())
    col_s, col_m = st.columns([2, 1])
    with col_s:
        selected = st.selectbox("Country", countries,
                                index=countries.index("China") if "China" in countries else 0)
    with col_m:
        mode = st.radio("Series", ["Cumulative", "Annual"], horizontal=True)

    data    = df[df["country"] == selected].sort_values("year")
    if len(data) < 4:
        st.warning("Insufficient data (minimum 4 years required)."); return

    t       = np.arange(len(data), dtype=float)
    S       = data["ev_sales"].cumsum().values
    y_ann   = data["ev_sales"].values
    horizon = st.slider("Projection horizon (years beyond last data point)", 2, 20, 8)
    t_all   = np.arange(len(data) + horizon, dtype=float)
    yrs_all = np.arange(int(data["year"].min()),
                         int(data["year"].min()) + len(t_all))

    # ── Fit models ────────────────────────────────────────────────────────
    models   = {"Bass": BassDiffusion(), "Logistic": LogisticGrowth(), "Gompertz": GompertzModel()}
    fitted   = {}
    diag     = {}   # diagnostics per model

    for name, model in models.items():
        try:
            model.fit(t, S)
            pred      = model.predict(t)
            resid     = S - pred
            ss_res    = np.sum(resid ** 2)
            ss_tot    = np.sum((S - S.mean()) ** 2)
            r2        = 1 - ss_res / ss_tot
            rmse      = np.sqrt(np.mean(resid ** 2))
            mape      = np.mean(np.abs(resid / (S + 1e-9))) * 100
            n, k      = len(S), len(model.get_params())
            aic       = n * np.log(ss_res / n) + 2 * k
            bic       = n * np.log(ss_res / n) + k * np.log(n)
            diag[name] = dict(r2=r2, rmse=rmse, mape=mape, aic=aic, bic=bic)
            fitted[name] = model
        except Exception:
            pass

    # ── Diagnostics ───────────────────────────────────────────────────────
    st.markdown("#### Model Diagnostics")
    if diag:
        # Rank by AIC (lower = better)
        best = min(diag, key=lambda m: diag[m]["aic"])
        param_cols = st.columns(3)
        for i, (name, model) in enumerate(models.items()):
            with param_cols[i]:
                if name in fitted:
                    d  = diag[name]
                    lbl = " 🥇" if name == best else ""
                    st.success(f"**{name}**{lbl}")
                    st.metric("R²",   f"{d['r2']:.4f}")
                    st.metric("RMSE", f"{d['rmse']:,.0f}")
                    st.metric("MAPE", f"{d['mape']:.1f}%")
                    st.metric("AIC",  f"{d['aic']:.1f}")
                    with st.expander("Parameters"):
                        for k, v in fitted[name].get_params().items():
                            st.caption(f"{k} = {v:.4f}")
                else:
                    st.error(f"**{name}** — did not converge")

    st.markdown("---")

    # ── Main chart ────────────────────────────────────────────────────────
    fig = go.Figure()
    if mode == "Cumulative":
        fig.add_trace(go.Scatter(
            x=data["year"].values, y=S / 1e3,
            mode="markers+lines", name="Observed",
            line=dict(color=SLATE, width=1.5, dash="dot"),
            marker=dict(size=6, color=SLATE, line=dict(width=1, color="white")),
        ))
        for name, model in fitted.items():
            fig.add_trace(go.Scatter(
                x=yrs_all, y=model.predict(t_all) / 1e3,
                mode="lines", name=f"{name} (R²={diag[name]['r2']:.3f})",
                line=dict(color=_COLORS[name], width=2, dash="dash"),
            ))
        ylab, title = "Thousand units (cumulative)", f"Cumulative Adoption — {selected}"
    else:
        fig.add_trace(go.Bar(
            x=data["year"].values, y=y_ann / 1e3,
            name="Observed", marker_color=SLATE, opacity=0.4,
        ))
        for name, model in fitted.items():
            pred_cum = model.predict(t_all)
            pred_ann = np.diff(pred_cum, prepend=pred_cum[0])
            fig.add_trace(go.Scatter(
                x=yrs_all, y=pred_ann / 1e3,
                mode="lines", name=f"{name} (R²={diag[name]['r2']:.3f})",
                line=dict(color=_COLORS[name], width=2, dash="dash"),
            ))
        ylab, title = "Thousand units (annual)", f"Annual Sales Modelled — {selected}"

    fig.add_vline(x=int(data["year"].max()), line_dash="dot",
                  line_color=SLATE, line_width=1,
                  annotation_text="Today",
                  annotation_font_color=SLATE, annotation_font_size=10)
    lay = base_layout(420, title)
    lay.update({"xaxis": {**lay.get("xaxis", {}), "title": "Year"},
                 "yaxis": {**lay.get("yaxis", {}), "title": ylab}})
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Projection table ──────────────────────────────────────────────────
    if fitted:
        st.markdown("#### Projections (future years)")
        future_t   = np.arange(len(data), len(data) + horizon, dtype=float)
        future_yrs = np.arange(int(data["year"].max()) + 1,
                                int(data["year"].max()) + 1 + horizon)
        rows = []
        for yr, ti in zip(future_yrs, future_t):
            row = {"Year": int(yr)}
            for name, model in fitted.items():
                row[f"{name} (k)"] = round(model.predict(np.array([ti]))[0] / 1e3, 1)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Multi-country comparison ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Multi-Country Bass Comparison")
    compare = st.multiselect(
        "Add countries",
        [c for c in countries if c != selected],
        default=["Norway", "Germany"][:2] if "Norway" in countries else [],
        max_selections=4,
    )
    if compare:
        palette = [NAVY, "#2e7d32", AMBER, "#8b1a1a", "#2e5090"]
        fig_cmp = go.Figure()
        for ci, ctry in enumerate([selected] + compare):
            d_c = df[df["country"] == ctry].sort_values("year")
            if len(d_c) < 4: continue
            t_c = np.arange(len(d_c), dtype=float)
            S_c = d_c["ev_sales"].cumsum().values
            fig_cmp.add_trace(go.Scatter(
                x=d_c["year"].values, y=S_c / 1e3, mode="markers",
                name=f"{ctry} (obs)",
                marker=dict(size=5, color=palette[ci % len(palette)]),
            ))
            try:
                bass = BassDiffusion(); bass.fit(t_c, S_c)
                t_ext = np.arange(0, len(t_c) + horizon, dtype=float)
                y_ext = np.arange(int(d_c["year"].min()),
                                   int(d_c["year"].min()) + len(t_ext))
                fig_cmp.add_trace(go.Scatter(
                    x=y_ext, y=bass.predict(t_ext) / 1e3, mode="lines",
                    name=f"{ctry} (Bass)",
                    line=dict(color=palette[ci % len(palette)], width=1.8, dash="dash"),
                ))
            except Exception:
                pass
        fig_cmp.add_vline(x=int(df["year"].max()), line_dash="dot",
                           line_color=SLATE, line_width=1)
        lay_c = base_layout(360)
        lay_c.update({"xaxis": {**lay_c.get("xaxis", {}), "title": "Year"},
                       "yaxis": {**lay_c.get("yaxis", {}),
                                  "title": "Thousand units (cumulative)"}})
        fig_cmp.update_layout(**lay_c)
        st.plotly_chart(fig_cmp, use_container_width=True, config=PLOTLY_CONFIG)