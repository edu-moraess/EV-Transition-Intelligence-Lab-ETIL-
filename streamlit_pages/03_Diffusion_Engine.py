# streamlit_pages/03_Diffusion_Engine.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel
from visualization.theme import chart_config, base_layout, NAVY, GREEN_DARK, RED_DARK, NAVY_LIGHT, AMBER, SLATE

_COLORS = {"Bass": NAVY, "Logistic": GREEN_DARK, "Gompertz": AMBER}


def render(df: pd.DataFrame):
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
    st.plotly_chart(fig, use_container_width=True, config=chart_config())

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
    st.markdown("### Multi-Country Comparison — Where is each country on the S-curve?")
    st.markdown(
        "Overlaying Bass curves across countries reveals who is ahead, who is catching up, "
        "and how much runway remains. The **normalised view** (% of market ceiling) "
        "removes the size advantage of large countries and shows adoption stage directly."
    )
    compare = st.multiselect(
        "Add countries to compare",
        [c for c in countries if c != selected],
        default=[c for c in ["Norway","Germany","Brazil","India"] if c in countries and c != selected][:3],
        max_selections=5,
    )

    all_compare = [selected] + (compare or [])
    palette = [NAVY, GREEN_DARK, AMBER, RED_DARK, NAVY_LIGHT, GOLD]

    tab_abs, tab_norm, tab_table = st.tabs(["Cumulative sales", "% of market ceiling", "Parameter table"])

    with tab_abs:
        fig_cmp = go.Figure()
        for ci, ctry in enumerate(all_compare):
            d_c = df[df["country"] == ctry].sort_values("year")
            if len(d_c) < 4: continue
            t_c = np.arange(len(d_c), dtype=float)
            S_c = d_c["ev_sales"].cumsum().values
            col = palette[ci % len(palette)]
            fig_cmp.add_trace(go.Scatter(
                x=d_c["year"].values, y=S_c / 1e3, mode="markers",
                name=f"{ctry} (observed)",
                marker=dict(size=5, color=col, opacity=0.7),
                showlegend=True,
            ))
            try:
                bass_c = BassDiffusion(); bass_c.fit(t_c, S_c)
                t_ext  = np.arange(0, len(t_c) + horizon, dtype=float)
                y_ext  = np.arange(int(d_c["year"].min()),
                                    int(d_c["year"].min()) + len(t_ext))
                fig_cmp.add_trace(go.Scatter(
                    x=y_ext, y=bass_c.predict(t_ext) / 1e3, mode="lines",
                    name=f"{ctry} (Bass fit)",
                    line=dict(color=col, width=2.2, dash="dash"),
                    showlegend=False,
                ))
            except Exception:
                pass
        fig_cmp.add_vline(x=int(df["year"].max()), line_dash="dot",
                           line_color=SLATE, line_width=1,
                           annotation_text="Today", annotation_font_size=9)
        lay_c = base_layout(400, "Cumulative EV Adoption — Bass Model Comparison")
        lay_c.update({"xaxis": {**lay_c.get("xaxis",{}), "title": "Year"},
                       "yaxis": {**lay_c.get("yaxis",{}), "title": "Thousand units (cumulative)"}})
        fig_cmp.update_layout(**lay_c)
        st.plotly_chart(fig_cmp, use_container_width=True, config=chart_config())

    with tab_norm:
        st.caption(
            "Normalised to % of fitted market ceiling (M). "
            "This removes scale and shows how far along the adoption curve each country is — "
            "0% = no EVs, 100% = full saturation."
        )
        fig_norm = go.Figure()
        for ci, ctry in enumerate(all_compare):
            d_c = df[df["country"] == ctry].sort_values("year")
            if len(d_c) < 4: continue
            t_c = np.arange(len(d_c), dtype=float)
            S_c = d_c["ev_sales"].cumsum().values
            col = palette[ci % len(palette)]
            try:
                bass_c = BassDiffusion(); bass_c.fit(t_c, S_c)
                M_c    = bass_c.get_params().get("M", 1)
                t_ext  = np.arange(0, len(t_c) + horizon, dtype=float)
                y_ext  = np.arange(int(d_c["year"].min()),
                                    int(d_c["year"].min()) + len(t_ext))
                pct    = bass_c.predict(t_ext) / M_c * 100
                # mark current position
                current_pct = S_c[-1] / M_c * 100
                fig_norm.add_trace(go.Scatter(
                    x=y_ext, y=pct, mode="lines",
                    name=f"{ctry} ({current_pct:.0f}% adopted)",
                    line=dict(color=col, width=2.5),
                    hovertemplate=f"{ctry} · %{{x}}: %{{y:.1f}}% of ceiling<extra></extra>",
                ))
                # annotate current position
                fig_norm.add_trace(go.Scatter(
                    x=[int(d_c["year"].max())], y=[current_pct],
                    mode="markers+text",
                    text=[f"{current_pct:.0f}%"],
                    textposition="top right",
                    textfont=dict(size=9, color=col),
                    marker=dict(size=9, color=col),
                    showlegend=False,
                ))
            except Exception:
                pass
        fig_norm.add_vline(x=int(df["year"].max()), line_dash="dot",
                            line_color=SLATE, line_width=1,
                            annotation_text="Today", annotation_font_size=9)
        fig_norm.add_hline(y=50, line_dash="dot",
                            line_color="rgba(128,128,128,0.4)", line_width=1,
                            annotation_text="50% saturation", annotation_font_size=9)
        lay_n = base_layout(400, "Adoption Stage — % of Fitted Market Ceiling")
        lay_n.update({"xaxis": {**lay_n.get("xaxis",{}), "title": "Year"},
                       "yaxis": {**lay_n.get("yaxis",{}), "title": "% of market ceiling",
                                   "range": [0, 105]}})
        fig_norm.update_layout(**lay_n)
        st.plotly_chart(fig_norm, use_container_width=True, config=chart_config())

    with tab_table:
        st.caption("Bass parameters fitted per country. M = ceiling, p = innovation rate, q = imitation rate.")
        table_rows = []
        for ctry in all_compare:
            d_c = df[df["country"] == ctry].sort_values("year")
            if len(d_c) < 4: continue
            t_c = np.arange(len(d_c), dtype=float)
            S_c = d_c["ev_sales"].cumsum().values
            try:
                bass_c = BassDiffusion(); bass_c.fit(t_c, S_c)
                params = bass_c.get_params()
                M_c = params.get("M", 0)
                current_pct = S_c[-1] / M_c * 100 if M_c > 0 else 0
                table_rows.append({
                    "Country":   ctry,
                    "M (ceiling)": f"{M_c/1e3:,.0f}k",
                    "p":          f"{params.get('p', 0):.4f}",
                    "q":          f"{params.get('q', 0):.4f}",
                    "% adopted":  f"{current_pct:.1f}%",
                    "Stage":      "Mature" if current_pct > 40 else
                                  "Acceleration" if current_pct > 15 else
                                  "Expansion" if current_pct > 5 else "Emerging",
                })
            except Exception:
                table_rows.append({"Country": ctry, "Stage": "Model failed"})
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

