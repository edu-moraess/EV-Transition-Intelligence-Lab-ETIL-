# streamlit_pages/02_Country_Lab.py
"""
Country Lab — quantitative deep-dive per country.

Sections:
  1. Key metrics (KPIs)
  2. EV sales history + YoY growth
  3. EV stock share trend
  4. Diffusion models (Bass / Logistic / Gompertz)
  5. Regime detection
  6. Fuel price context (IEA Energy Prices — where available)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.country_analysis import CountryAnalyzer
from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel
from data.fuel_loader import get_fuel_summary, get_monthly_fuel, covered_countries
from visualization.theme import (
    base_layout, bar_color_signed,
    NAVY, NAVY_LIGHT, RED_DARK, GREEN_MED, AMBER, SLATE
)

_DIFF_COLORS = {"Bass": NAVY, "Logistic": "#2e7d32", "Gompertz": AMBER}
_REGIME_COLORS = {0: SLATE, 1: NAVY_LIGHT, 2: GREEN_MED, 3: AMBER}
_REGIME_LABELS = {0: "Emerging", 1: "Expansion", 2: "Acceleration", 3: "Maturity"}


def render(df: pd.DataFrame):
    st.title("Country Lab")
    st.caption("Quantitative deep-dive — EV adoption, diffusion models and fuel price context")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    selected = st.selectbox("Country", countries,
                            index=countries.index("Norway") if "Norway" in countries else 0)
    if not selected:
        return

    try:
        analyzer = CountryAnalyzer(df, selected)
    except ValueError as e:
        st.error(str(e)); return

    data   = analyzer.get_data()
    latest = data.iloc[-1]

    # ── KPIs ─────────────────────────────────────────────────────────────
    gr   = latest.get("growth_rate", np.nan)
    cagr = latest.get("cagr", np.nan)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest year",  str(int(latest["year"])))
    c2.metric("EV sales",     f"{latest['ev_sales']:,.0f} units")
    c3.metric("YoY growth",   f"{gr:.1f}%" if not np.isnan(gr) else "N/A",
              delta=f"{gr:.1f}%" if not np.isnan(gr) else None)
    c4.metric("CAGR",         f"{cagr:.1f}%" if not np.isnan(cagr) else "N/A")

    # Fuel KPIs (if available)
    fuel_countries = covered_countries()
    has_fuel = selected in fuel_countries
    if has_fuel:
        fs = get_fuel_summary(selected)
        if fs:
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Gasoline price",
                      f"${fs['gasoline_latest']:.2f}/L" if fs["gasoline_latest"] else "N/A",
                      help=f"Latest: {fs['latest_date']}")
            f2.metric("Diesel price",
                      f"${fs['diesel_latest']:.2f}/L" if fs["diesel_latest"] else "N/A")
            f3.metric("Gasoline YoY",
                      f"{fs['gasoline_yoy_pct']:+.1f}%" if fs["gasoline_yoy_pct"] is not None else "N/A",
                      delta=f"{fs['gasoline_yoy_pct']:+.1f}%" if fs["gasoline_yoy_pct"] is not None else None)
            f4.metric("Fuel price volatility",
                      f"{fs['fuel_volatility']:.1f}%" if fs["fuel_volatility"] is not None else "N/A",
                      help="Annualised std dev of monthly gasoline returns (last 12 months)")

    st.markdown("---")

    # ── EV history + YoY ─────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("##### Annual EV Sales")
        fig = go.Figure(go.Bar(
            x=data["year"], y=data["ev_sales"] / 1e3,
            marker_color=NAVY,
            hovertemplate="%{x}: %{y:.1f}k units<extra></extra>",
        ))
        lay = base_layout(280)
        lay.update({"yaxis": {**lay.get("yaxis", {}), "title": "Thousand units"},
                     "showlegend": False})
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("##### YoY Growth (%)")
        gr_vals = data["growth_rate"].fillna(0)
        fig2 = go.Figure(go.Bar(
            x=data["year"], y=gr_vals,
            marker_color=bar_color_signed(gr_vals),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        lay2 = base_layout(280)
        lay2.update({"yaxis": {**lay2.get("yaxis", {}), "title": "%"},
                      "showlegend": False})
        fig2.update_layout(**lay2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── EV Stock Share ────────────────────────────────────────────────────
    if "ev_stock_share" in data.columns and data["ev_stock_share"].notna().any():
        st.markdown("##### EV Stock Share (%)")
        fig3 = go.Figure(go.Scatter(
            x=data["year"], y=data["ev_stock_share"],
            mode="lines+markers",
            line=dict(color=NAVY, width=2),
            marker=dict(size=5, color=NAVY),
            fill="tozeroy",
            fillcolor="rgba(31,58,110,0.07)",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        lay3 = base_layout(220)
        lay3.update({"yaxis": {**lay3.get("yaxis", {}), "title": "Share (%)"},
                      "showlegend": False})
        fig3.update_layout(**lay3)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Fuel price context ────────────────────────────────────────────────
    if has_fuel:
        st.markdown("---")
        st.markdown("##### Fuel Price Context — IEA Energy Prices (USD/litre)")
        st.caption(
            "Source: IEA Energy Prices Monthly Excerpt (Jun 2026). "
            "End-user prices including taxes."
        )

        monthly = get_monthly_fuel(selected)
        if not monthly.empty:
            # Dual-axis: EV sales (annual) left | Gasoline price (monthly) right
            fig_fuel = make_subplots(specs=[[{"secondary_y": True}]])

            # Gasoline monthly line
            gas = monthly.dropna(subset=["gasoline_usd"])
            fig_fuel.add_trace(go.Scatter(
                x=gas["date"], y=gas["gasoline_usd"],
                name="Gasoline (USD/L)",
                line=dict(color=AMBER, width=1.8),
                hovertemplate="%{x|%b %Y}: $%{y:.3f}/L<extra></extra>",
            ), secondary_y=False)

            # Diesel monthly line
            die = monthly.dropna(subset=["diesel_usd"])
            fig_fuel.add_trace(go.Scatter(
                x=die["date"], y=die["diesel_usd"],
                name="Diesel (USD/L)",
                line=dict(color=RED_DARK, width=1.2, dash="dot"),
                hovertemplate="%{x|%b %Y}: $%{y:.3f}/L<extra></extra>",
            ), secondary_y=False)

            # EV sales annual — bars on right axis
            ev_annual = data[["year", "ev_sales"]].copy()
            ev_annual["date"] = pd.to_datetime(ev_annual["year"].astype(str) + "-07-01")
            fig_fuel.add_trace(go.Bar(
                x=ev_annual["date"], y=ev_annual["ev_sales"] / 1e3,
                name="EV Sales (k units)",
                marker_color="rgba(31,58,110,0.25)",
                width=1000 * 3600 * 24 * 300,
                hovertemplate="%{x|%Y}: %{y:.1f}k EV units<extra></extra>",
            ), secondary_y=True)

            lay_f = base_layout(340)
            lay_f.update({
                "xaxis": {**lay_f.get("xaxis", {}), "title": ""},
                "yaxis": {**lay_f.get("yaxis", {}), "title": "Fuel price (USD/L)"},
            })
            fig_fuel.update_layout(**lay_f)
            fig_fuel.update_yaxes(title_text="EV Sales (thousand units)",
                                   secondary_y=True)
            st.plotly_chart(fig_fuel, use_container_width=True)

            # Insight note
            if fs.get("gasoline_yoy_pct") is not None:
                direction = "up" if fs["gasoline_yoy_pct"] > 0 else "down"
                st.caption(
                    f"Gasoline prices in {selected} are **{direction}** "
                    f"{abs(fs['gasoline_yoy_pct']):.1f}% year-on-year "
                    f"(latest: ${fs['gasoline_latest']:.2f}/L as of {fs['latest_date']}). "
                    f"Higher fuel costs increase the economic case for EV adoption."
                )
    elif not has_fuel:
        st.caption(
            f"ℹ️ Fuel price data not available for {selected} "
            f"(IEA coverage: {len(fuel_countries)} countries)."
        )

    st.markdown("---")

    # ── Diffusion models ──────────────────────────────────────────────────
    st.subheader("Diffusion Models")
    with st.spinner("Fitting models..."):
        results = analyzer.fit_diffusion_models()

    t   = np.arange(len(data), dtype=float)
    S   = data["ev_sales"].cumsum().values
    t_x = np.arange(0, len(t) + 8, dtype=float)
    y_x = np.arange(int(data["year"].min()),
                     int(data["year"].min()) + len(t_x))

    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(
        x=data["year"].values, y=S / 1e3,
        mode="markers+lines", name="Observed",
        line=dict(color=SLATE, width=1.5, dash="dot"),
        marker=dict(size=5, color=SLATE),
    ))

    param_cols = st.columns(3)
    for i, (name, model_cls) in enumerate(
        [("Bass", BassDiffusion()), ("Logistic", LogisticGrowth()), ("Gompertz", GompertzModel())]
    ):
        try:
            model_cls.fit(t, S)
            pred = model_cls.predict(t_x)
            r2   = 1 - np.sum((S - model_cls.predict(t)) ** 2) / np.sum((S - S.mean()) ** 2)
            fig_d.add_trace(go.Scatter(
                x=y_x, y=pred / 1e3, mode="lines",
                name=f"{name} (R²={r2:.3f})",
                line=dict(color=_DIFF_COLORS[name], width=2, dash="dash"),
            ))
            with param_cols[i]:
                st.success(f"**{name}** — R² = {r2:.4f}")
                for k, v in model_cls.get_params().items():
                    st.caption(f"{k} = {v:.4f}")
        except Exception:
            with param_cols[i]:
                st.warning(f"**{name}** — did not converge")

    fig_d.add_vline(x=int(data["year"].max()), line_dash="dot",
                    line_color=SLATE, line_width=1,
                    annotation_text="Today",
                    annotation_font_color=SLATE, annotation_font_size=10)
    lay_d = base_layout(360)
    lay_d.update({"xaxis": {**lay_d.get("xaxis", {}), "title": "Year"},
                   "yaxis": {**lay_d.get("yaxis", {}), "title": "Cumulative (thousand units)"}})
    fig_d.update_layout(**lay_d)
    st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("---")

    # ── Regime detection ──────────────────────────────────────────────────
    st.subheader("Adoption Regime Detection")
    with st.spinner("Detecting regimes..."):
        regimes = analyzer.detect_regimes()

    if "error" in regimes:
        st.warning(regimes["error"])
    else:
        states    = regimes["states"]
        reg_names = [_REGIME_LABELS.get(s, f"R{s}") for s in states]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=data["year"].values, y=data["ev_sales"].values / 1e3,
            mode="lines", line=dict(color=SLATE, width=1.5), name="EV Sales",
        ))
        for s in sorted(set(states)):
            mask = np.array(states) == s
            fig_r.add_trace(go.Scatter(
                x=data["year"].values[mask],
                y=data["ev_sales"].values[mask] / 1e3,
                mode="markers", name=_REGIME_LABELS.get(s, f"R{s}"),
                marker=dict(size=9, color=_REGIME_COLORS.get(s, SLATE),
                            line=dict(width=1, color="white")),
            ))
        lay_r = base_layout(300)
        lay_r.update({"xaxis": {**lay_r.get("xaxis", {}), "title": "Year"},
                       "yaxis": {**lay_r.get("yaxis", {}), "title": "Thousand units"}})
        fig_r.update_layout(**lay_r)
        st.plotly_chart(fig_r, use_container_width=True)

        st.dataframe(pd.DataFrame({
            "Year":       data["year"].values,
            "Regime":     reg_names,
            "EV Sales (k)": (data["ev_sales"].values / 1e3).round(1),
            "YoY (%)":    data["growth_rate"].fillna(0).round(1).values,
        }), use_container_width=True, hide_index=True)
