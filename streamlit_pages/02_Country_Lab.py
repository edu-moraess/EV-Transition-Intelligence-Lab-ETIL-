# streamlit_pages/02_Country_Lab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analytics.country_analysis import CountryAnalyzer
from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel
from data.fuel_loader import get_fuel_summary, get_monthly_fuel, covered_countries
from data.world_bank_client import get_country_fundamentals, get_fundamentals_score
from visualization.theme import (
    base_layout, bar_color_signed,
    NAVY, NAVY_LIGHT, RED_DARK, GREEN_MED, AMBER, SLATE
)

_DIFF_COLORS  = {"Bass": NAVY, "Logistic": "#2e7d32", "Gompertz": AMBER}
_REGIME_COLORS = {0: SLATE, 1: NAVY_LIGHT, 2: GREEN_MED, 3: AMBER}
_REGIME_LABELS = {0: "Emerging", 1: "Expansion", 2: "Acceleration", 3: "Maturity"}

_REGIME_EXPLAINER = {
    "Emerging":      "Early market — EV adoption is beginning. Policy and awareness are key.",
    "Expansion":     "Growing market — early adopters driving volume. Infrastructure expanding.",
    "Acceleration":  "Mass-market phase — mainstream buyers entering. Growth is rapid.",
    "Maturity":      "Mature market — high penetration. Growth is slowing naturally.",
}


def render(df: pd.DataFrame):
    st.markdown("## Country Lab")
    st.markdown("Deep-dive into any country's EV transition — history, trajectory, fuel context and regime.")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    selected  = st.selectbox("Select a country", countries,
                              index=countries.index("Germany") if "Germany" in countries else 0)
    if not selected:
        return

    try:
        analyzer = CountryAnalyzer(df, selected)
    except ValueError as e:
        st.error(str(e)); return

    data   = analyzer.get_data()
    latest = data.iloc[-1]
    gr     = latest.get("growth_rate", np.nan)
    cagr   = latest.get("cagr", np.nan)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EV sales (latest year)", f"{latest['ev_sales']:,.0f} units",
              help=f"Total EV units sold in {int(latest['year'])}.")
    c2.metric("Year-on-year growth",
              f"{gr:.1f}%" if not np.isnan(gr) else "N/A",
              delta=f"{gr:.1f}%" if not np.isnan(gr) else None,
              help="Change in EV sales vs the previous year.")
    c3.metric("CAGR (all years)",
              f"{cagr:.1f}%" if not np.isnan(cagr) else "N/A",
              help="Compound annual growth rate since the first data point.")
    if "ev_stock_share" in data.columns and data["ev_stock_share"].notna().any():
        latest_share = data["ev_stock_share"].dropna().iloc[-1]
        c4.metric("EV stock share",
                  f"{latest_share:.1f}%",
                  help="Share of EVs in the total vehicle fleet (most recent year).")
    else:
        c4.metric("EV stock share", "N/A")

    # Fuel KPIs
    fuel_countries = covered_countries()
    has_fuel = selected in fuel_countries
    if has_fuel:
        fs = get_fuel_summary(selected)
        if fs:
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Gasoline price (latest)",
                      f"${fs['gasoline_latest']:.2f}/L" if fs["gasoline_latest"] else "N/A",
                      help=f"End-user retail price including taxes · {fs['latest_date']}")
            f2.metric("Diesel price (latest)",
                      f"${fs['diesel_latest']:.2f}/L" if fs["diesel_latest"] else "N/A")
            f3.metric("Gasoline YoY change",
                      f"{fs['gasoline_yoy_pct']:+.1f}%" if fs["gasoline_yoy_pct"] is not None else "N/A",
                      delta=f"{fs['gasoline_yoy_pct']:+.1f}%" if fs["gasoline_yoy_pct"] is not None else None,
                      help="Year-on-year change in gasoline price. Rising prices strengthen the case for EVs.")
            f4.metric("Fuel price volatility",
                      f"{fs['fuel_volatility']:.1f}%" if fs["fuel_volatility"] is not None else "N/A",
                      help="Annualised std dev of monthly gasoline returns (last 12m). "
                           "High volatility increases the attractiveness of EVs as a stable-cost alternative.")

    # ── Country fundamentals (World Bank) ──────────────────────────────────────
    fundamentals = get_country_fundamentals(selected)
    if fundamentals.get("coverage", 0) > 0:
        st.markdown("---")
        st.markdown("### Economic context")
        st.markdown(
            "EV adoption doesn't happen in a vacuum — it depends on income levels, "
            "urbanization and economic momentum. These figures from the World Bank "
            "provide context for interpreting the EV trends above."
        )
        wb1, wb2, wb3, wb4, wb5 = st.columns(5)
        if "gdp_per_capita" in fundamentals:
            wb1.metric("GDP per capita",
                       f"${fundamentals['gdp_per_capita']:,.0f}",
                       help=f"World Bank, {fundamentals.get('gdp_per_capita_year','')}. "
                            "Higher income generally supports faster EV adoption.")
        if "population" in fundamentals:
            pop = fundamentals["population"]
            pop_label = f"{pop/1e6:.1f}M" if pop >= 1e6 else f"{pop:,.0f}"
            wb2.metric("Population", pop_label,
                       help=f"World Bank, {fundamentals.get('population_year','')}.")
        if "urbanization" in fundamentals:
            wb3.metric("Urbanization",
                       f"{fundamentals['urbanization']:.0f}%",
                       help=f"World Bank, {fundamentals.get('urbanization_year','')}. "
                            "Urban populations have shorter commutes and more charging access.")
        if "gdp_growth" in fundamentals:
            wb4.metric("GDP growth",
                       f"{fundamentals['gdp_growth']:+.1f}%",
                       help=f"World Bank, {fundamentals.get('gdp_growth_year','')}. "
                            "Economic momentum supports vehicle purchases generally.")
        score = get_fundamentals_score(fundamentals)
        if score is not None:
            wb5.metric("Fundamentals score",
                       f"{score:.0f}/100",
                       help="Equally-weighted composite of GDP per capita, urbanization, "
                            "GDP growth and CO2/capita — a context score, not a predictive factor.")

    st.markdown("---")

    # ── Sales + YoY ──────────────────────────────────────────────────────────
    st.markdown("### Sales history and growth")
    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure(go.Bar(
            x=data["year"], y=data["ev_sales"] / 1e3,
            marker_color=NAVY, opacity=0.85,
            hovertemplate="%{x}: %{y:.1f}k units<extra></extra>",
        ))
        lay = base_layout(280, "Annual EV Sales (thousand units)")
        lay.update({"yaxis": {**lay.get("yaxis",{}), "title": "k units"}, "showlegend": False})
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        gr_vals = data["growth_rate"].fillna(0)
        fig2 = go.Figure(go.Bar(
            x=data["year"], y=gr_vals,
            marker_color=bar_color_signed(gr_vals),
            hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
        ))
        lay2 = base_layout(280, "Year-on-Year Growth (%)")
        lay2.update({"yaxis": {**lay2.get("yaxis",{}), "title": "%"}, "showlegend": False})
        fig2.update_layout(**lay2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Stock share ───────────────────────────────────────────────────────────
    if "ev_stock_share" in data.columns and data["ev_stock_share"].notna().any():
        st.markdown(
            "**EV stock share** shows what percentage of all cars on the road are electric. "
            "It rises more slowly than sales — it takes years for the fleet to turn over."
        )
        fig3 = go.Figure(go.Scatter(
            x=data["year"], y=data["ev_stock_share"],
            mode="lines+markers",
            line=dict(color=NAVY, width=2.5),
            marker=dict(size=6, color=NAVY),
            fill="tozeroy", fillcolor="rgba(31,58,110,0.07)",
            hovertemplate="%{x}: %{y:.1f}% EV stock share<extra></extra>",
        ))
        lay3 = base_layout(220, "EV Stock Share (%)")
        lay3.update({"yaxis": {**lay3.get("yaxis",{}), "title": "% of fleet"}, "showlegend": False})
        fig3.update_layout(**lay3)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Fuel price context ────────────────────────────────────────────────────
    if has_fuel:
        st.markdown("### Fuel price context")
        st.markdown(
            "Higher and more volatile gasoline prices strengthen the economic case for switching to EVs. "
            "The chart below overlays monthly fuel prices with annual EV sales."
        )
        monthly = get_monthly_fuel(selected)
        if not monthly.empty:
            fig_fuel = make_subplots(specs=[[{"secondary_y": True}]])

            # EV sales bars added FIRST so they render behind the price lines,
            # and sized narrow + very transparent so they read as a subtle
            # backdrop rather than a solid block covering the chart.
            ev_annual = data[["year", "ev_sales"]].copy()
            ev_annual["date"] = pd.to_datetime(ev_annual["year"].astype(str) + "-07-01")
            fig_fuel.add_trace(go.Bar(
                x=ev_annual["date"], y=ev_annual["ev_sales"] / 1e3,
                name="EV Sales (k units, right axis)",
                marker_color=NAVY, opacity=0.18,
                width=1000 * 3600 * 24 * 60,  # ~60 days — a slim bar per year, not a solid block
                hovertemplate="%{x|%Y}: %{y:.1f}k EV units<extra></extra>",
            ), secondary_y=True)

            gas = monthly.dropna(subset=["gasoline_usd"])
            fig_fuel.add_trace(go.Scatter(
                x=gas["date"], y=gas["gasoline_usd"],
                name="Gasoline (USD/L)",
                line=dict(color=AMBER, width=2),
                hovertemplate="%{x|%b %Y}: $%{y:.3f}/L<extra></extra>",
            ), secondary_y=False)

            die = monthly.dropna(subset=["diesel_usd"])
            if not die.empty:
                fig_fuel.add_trace(go.Scatter(
                    x=die["date"], y=die["diesel_usd"],
                    name="Diesel (USD/L)",
                    line=dict(color=RED_DARK, width=1.5, dash="dot"),
                    hovertemplate="%{x|%b %Y}: $%{y:.3f}/L<extra></extra>",
                ), secondary_y=False)

            fig_fuel.update_layout(
                height=340, margin=dict(l=0, r=0, t=8, b=0),
                font=dict(family="Inter, Arial, sans-serif", size=11),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(128,128,128,0.2)",
                            borderwidth=1, font=dict(size=10), orientation="h",
                            yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor="rgba(128,128,128,0.12)", tickfont=dict(size=10)),
                yaxis=dict(title="Fuel price (USD/L)", gridcolor="rgba(128,128,128,0.12)"),
                yaxis2=dict(title="EV Sales (k units)", showgrid=False),
            )
            st.plotly_chart(fig_fuel, use_container_width=True)

            if fs.get("gasoline_yoy_pct") is not None:
                direction = "up" if fs["gasoline_yoy_pct"] > 0 else "down"
                color_txt = "🔴" if direction == "up" else "🟢"
                st.caption(
                    f"{color_txt} Gasoline in **{selected}** is **{direction}** "
                    f"{abs(fs['gasoline_yoy_pct']):.1f}% year-on-year "
                    f"(latest: ${fs['gasoline_latest']:.2f}/L, {fs['latest_date']}). "
                    f"{'Rising fuel costs increase the economic incentive to switch to EVs.' if direction == 'up' else 'Falling fuel prices reduce short-term financial pressure to adopt EVs.'}"
                )
    else:
        st.caption(
            f"ℹ️ Fuel price data not available for {selected}. "
            f"IEA Energy Prices covers {len(fuel_countries)} countries."
        )

    st.markdown("---")

    # ── Diffusion models ──────────────────────────────────────────────────────
    st.markdown("### Diffusion models — where is this market in the adoption curve?")
    st.markdown(
        "Technology adoption follows predictable S-shaped curves. "
        "Three models are fitted here: **Bass** (innovation + imitation dynamics), "
        "**Logistic** (symmetric S-curve) and **Gompertz** (asymmetric, slower early growth). "
        "The best-fitting model (lowest AIC) is highlighted."
    )

    t = np.arange(len(data), dtype=float)
    S = data["ev_sales"].cumsum().values
    t_x = np.arange(0, len(t) + 8, dtype=float)
    y_x = np.arange(int(data["year"].min()), int(data["year"].min()) + len(t_x))

    fitted = {}; diag = {}
    for name, model in [("Bass", BassDiffusion()), ("Logistic", LogisticGrowth()), ("Gompertz", GompertzModel())]:
        try:
            model.fit(t, S)
            pred   = model.predict(t)
            ss_res = np.sum((S - pred) ** 2)
            ss_tot = np.sum((S - S.mean()) ** 2)
            n, k   = len(S), len(model.get_params())
            r2     = 1 - ss_res / ss_tot
            aic    = n * np.log(ss_res / n) + 2 * k
            diag[name]   = dict(r2=r2, aic=aic)
            fitted[name] = model
        except Exception:
            pass

    best = min(diag, key=lambda m: diag[m]["aic"]) if diag else None

    param_cols = st.columns(3)
    for i, name in enumerate(["Bass", "Logistic", "Gompertz"]):
        with param_cols[i]:
            if name in fitted:
                lbl = " ✦ Best fit" if name == best else ""
                st.success(f"**{name}**{lbl}")
                st.caption(f"R² = {diag[name]['r2']:.3f} · AIC = {diag[name]['aic']:.0f}")
                for k, v in fitted[name].get_params().items():
                    st.caption(f"{k} = {v:.4f}")
            else:
                st.error(f"**{name}** — did not converge")

    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(
        x=data["year"].values, y=S / 1e3, mode="markers+lines", name="Observed",
        line=dict(color=SLATE, width=1.5, dash="dot"),
        marker=dict(size=5, color=SLATE),
    ))
    for name, model in fitted.items():
        fig_d.add_trace(go.Scatter(
            x=y_x, y=model.predict(t_x) / 1e3, mode="lines",
            name=f"{name} (R²={diag[name]['r2']:.3f})",
            line=dict(color=_DIFF_COLORS[name], width=2.5 if name == best else 1.8,
                      dash="solid" if name == best else "dash"),
        ))
    fig_d.add_vline(x=int(data["year"].max()), line_dash="dot",
                    line_color=SLATE, line_width=1,
                    annotation_text="Today", annotation_font_size=10)
    lay_d = base_layout(360, "Cumulative EV Adoption — Observed vs Model Fit")
    lay_d.update({"xaxis": {**lay_d.get("xaxis",{}), "title": "Year"},
                   "yaxis": {**lay_d.get("yaxis",{}), "title": "Thousand units (cumulative)"}})
    fig_d.update_layout(**lay_d)
    st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("---")

    # ── Regime detection ──────────────────────────────────────────────────────
    st.markdown("### Adoption regime — what stage is this market in?")
    st.markdown(
        "Markets go through distinct phases of EV adoption. "
        "A Gaussian Mixture Model identifies which phase each year belongs to based on "
        "EV stock share and growth rate patterns."
    )

    with st.spinner("Detecting regimes..."):
        regimes = analyzer.detect_regimes()

    if "error" in regimes:
        st.warning(regimes["error"])
    else:
        states    = regimes["states"]
        reg_names = [_REGIME_LABELS.get(s, f"R{s}") for s in states]
        current_regime = reg_names[-1] if reg_names else "Unknown"

        # Show current regime with explanation
        regime_col, explain_col = st.columns([1, 2])
        with regime_col:
            regime_colors = {"Emerging": "🔵", "Expansion": "🟢", "Acceleration": "🟡", "Maturity": "🔴"}
            icon = regime_colors.get(current_regime, "⚪")
            st.metric("Current regime", f"{icon} {current_regime}")
        with explain_col:
            st.info(_REGIME_EXPLAINER.get(current_regime, ""))

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=data["year"].values, y=data["ev_sales"].values / 1e3,
            mode="lines", line=dict(color=SLATE, width=1.5), name="EV Sales",
        ))
        for s in sorted(set(states)):
            mask = np.array(states) == s
            fig_r.add_trace(go.Scatter(
                x=data["year"].values[mask], y=data["ev_sales"].values[mask] / 1e3,
                mode="markers", name=_REGIME_LABELS.get(s, f"R{s}"),
                marker=dict(size=10, color=_REGIME_COLORS.get(s, SLATE),
                            line=dict(width=1.5, color="white")),
            ))
        lay_r = base_layout(300, f"EV Adoption Regimes — {selected}")
        lay_r.update({"xaxis": {**lay_r.get("xaxis",{}), "title": "Year"},
                       "yaxis": {**lay_r.get("yaxis",{}), "title": "Thousand units"}})
        fig_r.update_layout(**lay_r)
        st.plotly_chart(fig_r, use_container_width=True)
