# streamlit_pages/14_Scenario_Simulator.py
"""Scenario Simulator page."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.advanced_analytics import project_scenario
import numpy as np

def render(df: pd.DataFrame):
    st.title("🔮 Scenario Simulator")
    st.markdown("Adjust macroeconomic levers to project future EV adoption.")

    if df.empty:
        st.warning("No data loaded.")
        return

    base_year = df["year"].max()
    st.sidebar.markdown("### Sliders (relative changes)")

    oil_price = st.sidebar.slider("Oil Price Change (%)", -50, 50, 0, 5)
    lithium_price = st.sidebar.slider("Lithium Price Change (%)", -30, 30, 0, 5)
    interest_rate = st.sidebar.slider("Interest Rate Change (pp)", -2.0, 2.0, 0.0, 0.25)
    incentives = st.sidebar.slider("Government Incentive Change (%)", -50, 50, 0, 5)
    carbon_tax = st.sidebar.slider("Carbon Tax Change (%)", -20, 50, 0, 5)
    gdp_growth = st.sidebar.slider("GDP Growth Change (%)", -5, 10, 0, 1)

    adjustments = {
        "oil_price_change": oil_price,
        "lithium_price_change": lithium_price,
        "interest_rate_change": interest_rate,
        "incentive_change": incentives,
        "carbon_tax_change": carbon_tax,
        "gdp_growth_change": gdp_growth,
    }

    try:
        projected = project_scenario(df, adjustments)
    except Exception as e:
        st.error(f"Projection error: {e}")
        return

    # Comparison
    base = df[df["year"] == base_year]
    st.subheader(f"Impact on Global EV Sales ({base_year})")
    col1, col2 = st.columns(2)
    total_base = base["ev_sales"].sum()
    total_proj = projected["ev_sales"].sum()
    col1.metric("Base Sales", f"{total_base:,.0f}")
    col2.metric("Projected Sales", f"{total_proj:,.0f}", delta=f"{total_proj-total_base:+,.0f}")

    # Top 10 countries
    top_countries = base.groupby("country")["ev_sales"].sum().nlargest(5).index
    comp_df = base[base["country"].isin(top_countries)][["country", "ev_sales"]].merge(
        projected[["country", "ev_sales"]], on="country", suffixes=("_base", "_proj")
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comp_df["country"], y=comp_df["ev_sales_base"], name="Base"))
    fig.add_trace(go.Bar(x=comp_df["country"], y=comp_df["ev_sales_proj"], name="Projected"))
    fig.update_layout(barmode="group", title="Top 5 Countries Comparison")
    st.plotly_chart(fig, use_container_width=True)

    # Download projected data
    csv = projected.to_csv(index=False).encode('utf-8')
    st.download_button("Download Projections", csv, "scenario_projection.csv", "text/csv")