# streamlit_pages/09_Scenarios.py
"""
Scenario Engine — What if? analysis for EV policy and infrastructure.

Uses the Bass diffusion model to simulate how changes in key drivers
(charging infrastructure, subsidy levels, innovation rate) would alter
a country's adoption trajectory. Built on the existing ScenarioEngine
in analytics/scenario_engine.py.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from analytics.scenario_engine import ScenarioEngine
from visualization.theme import (
    base_layout, chart_config,
    NAVY, GREEN_MED, RED_DARK, SLATE,
)


def render(df: pd.DataFrame, embedded: bool = False):
    if not embedded:
        st.markdown("## Scenario Engine — What If?")
    st.markdown(
        "Point forecasts are too certain. Real policy decisions live in a space of "
        "possible futures — not a single line. This tool lets you define alternative "
        "scenarios by adjusting the drivers of EV adoption, and shows you how the "
        "diffusion trajectory changes under each one."
    )
    st.markdown(
        "Built on the **Bass diffusion model**: "
        "***p*** = innovation coefficient (early adopters, policy pull), "
        "***q*** = imitation coefficient (word of mouth, network effects), "
        "***M*** = market potential (long-run ceiling, shaped by infrastructure and income)."
    )
    st.markdown("---")

    countries = sorted(df["country"].unique())
    col_sel, col_horizon = st.columns([3, 1])
    with col_sel:
        selected = st.selectbox("Select a country", countries,
                                 index=countries.index("Brazil") if "Brazil" in countries else 0)
    with col_horizon:
        horizon = st.number_input("Forecast years", min_value=3, max_value=20, value=10)

    st.markdown("---")

    # ── Scenario builder ──────────────────────────────────────────────────────
    st.markdown("### Define your scenarios")
    
    # FIX: encapsular texto em container para evitar vazamento entre colunas
    with st.container():
        st.markdown(
            "Adjust the sliders to define up to 3 alternative scenarios. "
            "Each parameter adjusts the Bass model coefficient by the percentage you set."
        )

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown("##### 🟦 Baseline")
        st.caption("Current trajectory — no policy change")

    with col_s2:
        st.markdown("##### 🟡 Optimistic")
        p_opt = st.slider("Innovation rate (p) +%", 0, 100, 30, key="p_opt",
                          help="Higher p = stronger early adopter pull (subsidies, mandates)")
        q_opt = st.slider("Imitation rate (q) +%", 0, 100, 20, key="q_opt",
                          help="Higher q = stronger network effects (more EVs visible on road)")
        m_opt = st.slider("Market ceiling (M) +%", 0, 100, 40, key="m_opt",
                          help="Higher M = more infrastructure, higher income, lower EV prices")

    with col_s3:
        st.markdown("##### Conservative")
        p_con = st.slider("Innovation rate (p) -%", 0, 50, 15, key="p_con")
        q_con = st.slider("Imitation rate (q) -%", 0, 50, 10, key="q_con")
        m_con = st.slider("Market ceiling (M) -%", 0, 50, 20, key="m_con")

    st.markdown("---")

    # ── Run scenarios ─────────────────────────────────────────────────────────
    try:
        engine = ScenarioEngine(df, selected)

        baseline = engine.run_scenario("Baseline", {}, horizon=horizon)
        optimistic = engine.run_scenario("Optimistic", {
            "p": p_opt / 100, "q": q_opt / 100, "M": m_opt / 100,
        }, horizon=horizon)
        conservative = engine.run_scenario("Conservative", {
            "p": -p_con / 100, "q": -q_con / 100, "M": -m_con / 100,
        }, horizon=horizon)

    except Exception as e:
        st.error(f"Scenario engine error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    # ── Chart: annual sales ───────────────────────────────────────────────────
    st.markdown(f"### Annual EV sales projections — {selected}")
    st.caption(
        "Annual (not cumulative) EV sales under each scenario. "
        "The shaded area between optimistic and conservative is the uncertainty band."
    )

    fig1 = go.Figure()

    # Historical observed
    hist = df[df["country"] == selected].sort_values("year")
    if hist.empty:
        st.warning(f"No historical data available for {selected}.")
        return

    hist_years = hist["year"].values
    hist_sales = hist["ev_sales"].values
    fig1.add_trace(go.Scatter(
        x=hist_years, y=hist_sales / 1e3, mode="markers+lines",
        name="Observed (historical)",
        line=dict(color=SLATE, width=2, dash="dot"),
        marker=dict(size=6, color=SLATE),
        hovertemplate="%{x}: %{y:.1f}k observed<extra></extra>",
    ))

    colors = {"Baseline": NAVY, "Optimistic": GREEN_MED, "Conservative": RED_DARK}
    for scenario_name, scenario_df in [
        ("Baseline", baseline), ("Optimistic", optimistic), ("Conservative", conservative)
    ]:
        if scenario_df is None or scenario_df.empty:
            continue
        sales_col = "annual_sales" if "annual_sales" in scenario_df.columns else "sales"
        if sales_col not in scenario_df.columns:
            continue
        fig1.add_trace(go.Scatter(
            x=scenario_df["year"], y=scenario_df[sales_col] / 1e3,
            name=scenario_name, mode="lines",
            line=dict(color=colors[scenario_name],
                      width=2.5,
                      dash="dash" if scenario_name == "Conservative" else "solid"),
            hovertemplate=f"{scenario_name} · %{{x}}: %{{y:.1f}}k<extra></extra>",
        ))

    # Uncertainty band (mais visível)
    if (optimistic is not None and not optimistic.empty and
        conservative is not None and not conservative.empty):
        opt_col = "annual_sales" if "annual_sales" in optimistic.columns else "sales"
        con_col = "annual_sales" if "annual_sales" in conservative.columns else "sales"
        if opt_col in optimistic.columns and con_col in conservative.columns:
            merged_uc = optimistic[["year", opt_col]].merge(
                conservative[["year", con_col]], on="year", suffixes=("_opt", "_con")
            )
            future_uc = merged_uc[merged_uc["year"] > hist_years.max()]
            if not future_uc.empty:
                fig1.add_trace(go.Scatter(
                    x=pd.concat([future_uc["year"], future_uc["year"][::-1]]),
                    y=pd.concat([future_uc[opt_col + "_opt"] / 1e3,
                                  future_uc[con_col + "_con"][::-1] / 1e3]),
                    fill="toself",
                    fillcolor="rgba(31,81,209,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Uncertainty band",
                    hoverinfo="skip",
                ))

    # Today line
    fig1.add_vline(x=hist_years.max(), line_dash="dot",
                   line_color=SLATE, line_width=1,
                   annotation_text="Today", annotation_font_size=9)

    lay1 = base_layout(420, f"Annual EV Sales — {selected} — Scenario Comparison")
    lay1.update({
        "xaxis": {**lay1.get("xaxis", {}), "title": "Year"},
        "yaxis": {**lay1.get("yaxis", {}), "title": "Thousand units / year", "rangemode": "tozero"}
    })
    fig1.update_layout(**lay1)
    st.plotly_chart(fig1, config=chart_config())

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown(f"### {horizon}-year summary")
    rows = []
    for name, sdf in [("Baseline", baseline),
                       ("Optimistic", optimistic),
                       ("Conservative", conservative)]:
        if sdf is None or sdf.empty:
            continue
        sales_col = "annual_sales" if "annual_sales" in sdf.columns else "sales"
        if sales_col not in sdf.columns:
            continue

        # FIX: Peak usa o cenário COMPLETO (histórico + projeção)
        peak = sdf[sales_col].max()

        # Future-only para final year e total (apenas projeção)
        future_only = sdf[sdf["year"] > hist_years.max()]
        if future_only.empty:
            continue
        final = future_only[sales_col].iloc[-1]
        total = future_only[sales_col].sum()

        rows.append({
            "Scenario": name,
            "Peak sales": f"{peak / 1e3:.0f}k",
            f"Sales in {int(future_only['year'].iloc[-1])}": f"{final / 1e3:.0f}k",
            f"Total ({horizon}yr)": f"{total / 1e6:.2f}M",
            "vs Baseline": "—" if name == "Baseline" else "",
        })
    if rows:
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, hide_index=True)

    with st.expander("How are scenarios built?"):
        st.markdown("""
**Model:** Bass diffusion fitted on the observed annual sales time series.

**Parameters adjusted per scenario:**

| Parameter | What it represents | Policy levers |
|---|---|---|
| **p** (innovation) | Fraction of non-adopters who try the product independently | Subsidies, mandates, advertising |
| **q** (imitation) | Fraction who adopt after seeing others | Network effects, visibility, peer influence |
| **M** (ceiling) | Total addressable market | Infrastructure density, income, EV model availability |

**Scenario construction:**
- Baseline: fitted p, q, M unchanged
- Optimistic: p × (1 + slider%), q × (1 + slider%), M × (1 + slider%)
- Conservative: p × (1 − slider%), q × (1 − slider%), M × (1 − slider%)

**Limitation:** Bass diffusion is a reduced-form model. It does not endogenously model
supply constraints, competitor dynamics, or abrupt regulatory changes. Treat scenarios
as directional, not as point forecasts.
        """)
