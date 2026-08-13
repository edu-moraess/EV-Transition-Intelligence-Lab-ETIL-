# streamlit_pages/11_Research_Intelligence.py
"""
Research Intelligence — Integrated Macro-Quant EV Analysis

This page integrates all analytical layers into a unified research view:

  Pipeline: data → FactorEngine → RegimeAwareForecast → NarrativeEngine

  1. Multi-Factor Dashboard  — FactorEngine ranking + driver decomposition
  2. Regime-Aware Forecast   — conditional projections by adoption phase
  3. Research Narratives     — automated structured insights per country

This is the culmination of the ETIL architecture: quantitative outputs
translated into institutional research language, with full explainability.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models.scoring.factor_engine import FactorEngine
from models.forecasting.regime_forecast import RegimeAwareForecastEngine
from analytics.narrative_engine import NarrativeEngine
from ingestion.loader import IEALoader
from data.fuel_loader import load_fuel_prices
from visualization.theme import (
    base_layout, chart_config, bar_color_signed,
    NAVY, NAVY_LIGHT, NAVY_PALE, GREEN_MED, GREEN_DARK,
    AMBER, GOLD, RED_DARK, SLATE, QUINTIL_COLORS,
)

_REGIME_COLORS = {
    "Emerging":     RED_DARK,
    "Expansion":    AMBER,
    "Acceleration": GREEN_MED,
    "Maturity":     NAVY,
}


@st.cache_data(ttl=600, show_spinner="Running multi-factor engine...")
def _run_factor_engine(dataset: str, mode: str, _df: pd.DataFrame) -> dict:
    try:
        loader   = IEALoader()
        year_int = 2026 if "2026" in dataset or "combined" in dataset.lower() else 2025
        infra_df = loader.get_aggregator(year_int).ev_per_charger()
        fuel_df  = load_fuel_prices()
        fe = FactorEngine()
        fe.fit(_df, fuel_df=fuel_df, infra_df=infra_df)
        return {
            "ok":       True,
            "engine":   fe,
            "ranking":  fe.get_ranking(),
            "ic_table": fe.get_factor_ic(),
            "infra_df": infra_df,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@st.cache_data(ttl=600, show_spinner="Running regime forecasts...")
def _run_regime_forecasts(dataset: str, countries: tuple, _df: pd.DataFrame) -> dict:
    rfm     = RegimeAwareForecastEngine(n_simulations=500)
    results = {}
    for c in countries:
        c_df = _df[_df["country"] == c].sort_values("year")
        if len(c_df) < 4:
            continue
        r = rfm.forecast(c_df, horizon=10)
        if "error" not in r:
            results[c] = r
    return results


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "")
    mode    = df.attrs.get("vehicle_mode", "Cars")

    st.markdown("## Research Intelligence")
    st.markdown(
        "An integrated quantitative research view — factor scores, regime-aware forecasts "
        "and automated narratives, all derived from the same data pipeline with no manual overrides."
    )
    st.caption(
        "Pipeline: IEA data → FeatureEngineer → **FactorEngine** → "
        "**RegimeAwareForecast** → **NarrativeEngine**"
    )
    st.markdown("---")

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🎯 Multi-Factor Rankings",
        "📈 Regime-Aware Forecast",
        "📝 Research Narratives",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — MULTI-FACTOR RANKINGS
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Multi-Factor EV Transition Score")
        st.markdown(
            "The **FactorEngine** extends EVQTI from 2 factors to a full multi-dimensional model, "
            "combining penetration signal, energy cost pressure, infrastructure gap and macroeconomic "
            "capacity — all lagged t-1 and calibrated by empirical IC."
        )

        with st.spinner("Computing factor scores..."):
            fe_result = _run_factor_engine(dataset, mode, df)

        if not fe_result["ok"]:
            st.error(f"FactorEngine error: {fe_result.get('error')}")
            return

        fe      = fe_result["engine"]
        ranking = fe_result["ranking"]
        ic_tbl  = fe_result["ic_table"]

        # Factor IC validation
        if not ic_tbl.empty:
            st.markdown("#### Factor validation — which signals are predictive?")
            c1, c2, c3, c4 = st.columns(4)
            for i, (_, row) in enumerate(ic_tbl.head(4).iterrows()):
                col = [c1, c2, c3, c4][i]
                col.metric(
                    row["factor"].replace("_"," ").title(),
                    f"IC {row['mean_ic']:+.3f}",
                    f"{row['pct_sig']:.0%} years significant",
                    help=f"Group: {row['factor_group']} · Weight: {row.get('abs_ic',0)/ic_tbl['abs_ic'].sum():.1%}"
                )
            st.markdown("---")

        # Ranking chart
        n_show = st.slider("Countries to display", 10, len(ranking), min(25, len(ranking)))
        top_n  = ranking.head(n_show).sort_values("composite_score")
        norm   = (top_n["composite_score"] - top_n["composite_score"].min()) / \
                 (top_n["composite_score"].max() - top_n["composite_score"].min() + 1e-9)

        fig_r = go.Figure(go.Bar(
            x=top_n["composite_score"], y=top_n["country"], orientation="h",
            marker=dict(color=norm, colorscale=[[0,NAVY_PALE],[0.5,NAVY_LIGHT],[1,NAVY]],
                        showscale=False),
            text=top_n["composite_score"].map(lambda v: f"{v:.1f}"),
            textposition="outside", textfont=dict(size=9),
            hovertemplate="%{y}: %{x:.1f} composite score<extra></extra>",
        ))
        lay_r = base_layout(max(380, n_show * 22), f"Multi-Factor Score — Top {n_show}")
        lay_r.update({"xaxis": {**lay_r.get("xaxis",{}), "title": "Composite Score (0–100)"},
                       "yaxis": {**lay_r.get("yaxis",{}), "title": ""},
                       "showlegend": False, "margin": {**lay_r.get("margin",{}), "r": 60}})
        fig_r.update_layout(**lay_r)
        st.plotly_chart(fig_r, use_container_width=True, config=chart_config())

        # Driver decomposition for selected country
        st.markdown("#### Factor decomposition — what is driving each country's score?")
        countries_ranked = ranking["country"].tolist()
        selected_c = st.selectbox("Select a country", countries_ranked,
                                   index=min(5, len(countries_ranked)-1))
        drivers = fe.get_drivers(selected_c)
        if drivers:
            d1, d2, d3 = st.columns(3)
            d1.metric("Composite rank", f"#{drivers.get('rank','N/A')}")
            d2.metric("Score", f"{drivers.get('score', 0):.1f}/100")
            d3.metric("Factors used", str(drivers.get("n_factors", 0)))
            st.caption(f"**{drivers.get('headline','')}**")

            all_factors = (
                drivers.get("top_positive",[]) + drivers.get("top_negative",[])
            )
            if all_factors:
                fdf = pd.DataFrame(all_factors)
                fdf["label"] = fdf["factor"].str.replace("_"," ").str.title()
                fdf["color"] = fdf["contribution"].apply(
                    lambda v: GREEN_MED if v > 0 else RED_DARK
                )
                fig_d = go.Figure(go.Bar(
                    x=fdf["contribution"], y=fdf["label"], orientation="h",
                    marker_color=fdf["color"].tolist(),
                    text=fdf["contribution"].map(lambda v: f"{v:+.3f}"),
                    textposition="outside", textfont=dict(size=9),
                ))
                lay_d = base_layout(max(200, len(fdf)*40), f"Factor Contributions — {selected_c}")
                lay_d.update({"xaxis": {**lay_d.get("xaxis",{}), "title": "Contribution to score"},
                               "showlegend": False})
                fig_d.update_layout(**lay_d)
                st.plotly_chart(fig_d, use_container_width=True, config=chart_config())

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — REGIME-AWARE FORECAST
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### Regime-Aware Probabilistic Forecast")
        st.markdown(
            "The forecast is conditioned on the **current adoption regime** — "
            "Emerging, Expansion, Acceleration or Maturity — which determines the "
            "Bass diffusion priors and Monte Carlo volatility used in the simulation."
        )

        countries = sorted(df["country"].unique())
        col_c, col_h = st.columns([3, 1])
        with col_c:
            fc_country = st.selectbox("Country", countries,
                                       index=countries.index("Germany") if "Germany" in countries else 0,
                                       key="fc_country")
        with col_h:
            fc_horizon = st.number_input("Horizon (years)", 5, 15, 10, key="fc_horizon")

        with st.spinner(f"Simulating {fc_country} under all regimes..."):
            fc_results = _run_regime_forecasts(
                dataset, (fc_country,), df
            )

        if fc_country not in fc_results:
            st.warning("Insufficient data for this country.")
            st.stop()

        result  = fc_results[fc_country]
        regime  = result["regime_name"]
        bl      = result["baseline"]
        prior   = result["prior_applied"]
        narr    = result["narrative"]
        params  = result["bass_params"]

        # Regime summary
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Detected regime", regime,
                   help="Based on current EV stock share and growth rate dynamics.")
        r2.metric("q multiplier", f"{prior['q_mult']:.2f}×",
                   help="Imitation coefficient adjustment for this regime.")
        r3.metric("M ceiling range", f"{prior['M_range'][0]:.0%}–{prior['M_range'][1]:.0%}",
                   help="Fraction of fitted M sampled in Monte Carlo.")
        r4.metric("Forecast volatility", prior["volatility"],
                   help="Std dev of multiplicative noise in each simulation path.")

        st.markdown(f"*{narr['what']}*")
        st.markdown(f"**Why:** {narr['why']}")
        st.markdown("---")

        # Baseline forecast chart
        hist = df[df["country"] == fc_country].sort_values("year")
        fig_f = go.Figure()

        # Historical
        fig_f.add_trace(go.Scatter(
            x=hist["year"], y=hist["ev_sales"] / 1e3, mode="markers+lines",
            name="Observed", line=dict(color=SLATE, width=2, dash="dot"),
            marker=dict(size=6, color=SLATE),
        ))

        # 95% CI
        fig_f.add_trace(go.Scatter(
            x=pd.concat([bl["year"], bl["year"][::-1]]),
            y=pd.concat([bl["p95"] / 1e3, bl["p5"][::-1] / 1e3]),
            fill="toself", fillcolor="rgba(21,81,209,0.06)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        ))
        # 80% CI
        fig_f.add_trace(go.Scatter(
            x=pd.concat([bl["year"], bl["year"][::-1]]),
            y=pd.concat([bl["p90"] / 1e3, bl["p10"][::-1] / 1e3]),
            fill="toself", fillcolor="rgba(21,81,209,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="80% CI",
        ))
        # Median
        fig_f.add_trace(go.Scatter(
            x=bl["year"], y=bl["median"] / 1e3, mode="lines",
            name=f"Median ({regime})",
            line=dict(color=NAVY, width=2.5),
        ))

        # All-regime comparison
        show_regimes = st.checkbox("Show all-regime comparison", value=True)
        if show_regimes:
            for rid, rname in [(0,"Emerging"),(1,"Expansion"),(2,"Acceleration"),(3,"Maturity")]:
                if rid == result["regime_id"]: continue
                alt = result["all_regimes"].get(rid, pd.DataFrame())
                if alt.empty: continue
                fig_f.add_trace(go.Scatter(
                    x=alt["year"], y=alt["median"] / 1e3, mode="lines",
                    name=f"{rname} (alt)",
                    line=dict(color=_REGIME_COLORS.get(rname, SLATE), width=1.5, dash="dash"),
                    opacity=0.55,
                ))

        fig_f.add_vline(x=int(hist["year"].max()), line_dash="dot",
                         line_color=SLATE, line_width=1,
                         annotation_text="Today", annotation_font_size=9)

        lay_f = base_layout(420, f"Regime-Aware Forecast — {fc_country} ({regime})")
        lay_f.update({"xaxis": {**lay_f.get("xaxis",{}), "title": "Year"},
                       "yaxis": {**lay_f.get("yaxis",{}), "title": "Thousand EV units / year"}})
        fig_f.update_layout(**lay_f)
        st.plotly_chart(fig_f, use_container_width=True, config=chart_config())
        st.caption(f"**Implication:** {narr['implication']}")

        # Percentile table
        with st.expander("Forecast percentile table"):
            disp = bl.copy()
            for col in ["median","p10","p90","p5","p95","mean"]:
                disp[col] = (disp[col] / 1e3).round(1)
            disp.columns = ["Year","Median (k)","P10","P90","P5","P95","Mean"]
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — RESEARCH NARRATIVES
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### Automated Research Narratives")
        st.markdown(
            "Quantitative insights translated into structured research bullets. "
            "Every bullet is derived from a specific quantitative condition — "
            "no LLM hallucination, no generic text."
        )

        col_mode, _ = st.columns([2, 3])
        with col_mode:
            view = st.radio("View", ["Single country", "Global intelligence"],
                             horizontal=True)

        # Build narrative engine with full context
        infra_df = fe_result.get("infra_df", pd.DataFrame())
        fe_engine = fe_result.get("engine")

        # Pre-compute regime results for top countries
        top_countries = ranking["country"].head(15).tolist()
        with st.spinner("Building regime context..."):
            rfm = RegimeAwareForecastEngine(n_simulations=200)
            regime_ctx = {}
            for c in top_countries:
                c_df = df[df["country"]==c].sort_values("year")
                if len(c_df) < 4: continue
                r = rfm.forecast(c_df, horizon=5)
                if "error" not in r:
                    regime_ctx[c] = {
                        "current_regime": r["regime_name"],
                        "regime_change":  False,
                    }

        ne = NarrativeEngine(
            factor_engine=fe_engine,
            regime_results=regime_ctx,
            infra_df=infra_df,
        )

        if view == "Single country":
            narr_country = st.selectbox(
                "Country",
                sorted(df["country"].unique()),
                index=sorted(df["country"].unique()).index("Germany")
                if "Germany" in df["country"].unique() else 0,
                key="narr_country"
            )
            with st.spinner(f"Generating narrative for {narr_country}..."):
                narrative = ne.generate_country(narr_country, df)

            st.markdown(f"### {narrative['headline']}")
            st.markdown(f"**Year:** {narrative['year']} · "
                         f"**Regime:** {regime_ctx.get(narr_country, {}).get('current_regime','—')}")
            st.markdown("---")

            for bullet in narrative["bullets"]:
                st.markdown(bullet)

            if narrative["anomalies"]:
                st.markdown("---")
                st.markdown("#### ⚠️ Anomalies flagged")
                for a in narrative["anomalies"]:
                    st.warning(a)

            with st.expander("Structured output"):
                st.markdown(f"**What changed:** {narrative['what_changed']}")
                st.markdown(f"**Why:** {narrative['why']}")
                st.markdown(f"**Implication:** {narrative['implication']}")

        else:
            with st.spinner("Generating global intelligence..."):
                global_narr = ne.generate_global(df)

            st.markdown(f"### {global_narr['headline']}")
            st.markdown(f"**{global_narr['n_countries']} countries tracked · {global_narr['year']}**")
            st.markdown("---")

            for bullet in global_narr["bullets"]:
                st.markdown(bullet)

            if global_narr["anomalies"]:
                st.markdown("---")
                st.markdown("#### Countries flagged for review")
                cols = st.columns(min(3, len(global_narr["anomalies"])))
                for i, country in enumerate(global_narr["anomalies"]):
                    cols[i % 3].warning(country)

            st.markdown("---")
            st.markdown("#### Top 10 — Multi-Factor Score")
            top10 = ranking.head(10)[["country","composite_score","n_factors_used"]].copy()
            top10.columns = ["Country","Score","Factors used"]
            st.dataframe(top10, use_container_width=True, hide_index=True)
