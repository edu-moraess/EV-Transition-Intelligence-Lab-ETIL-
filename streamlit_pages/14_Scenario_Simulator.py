# streamlit_pages/14_Scenario_Simulator.py
"""Scenario Simulator — Professional Edition."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.advanced_analytics import project_scenario
import numpy as np

def render(df: pd.DataFrame):
    st.title("🔮 Scenario Simulator")
    st.markdown("Adjust macroeconomic levers to project the impact on EV sales, stock and penetration. Results are relative to the latest data year.")

    if df.empty:
        st.warning("No data loaded.")
        return

    base_year = df["year"].max()

    # Sidebar controls
    st.sidebar.markdown("### ⚙️ Scenario Settings")
    oil_price = st.sidebar.slider("Oil Price Change (%)", -50, 50, 0, 5)
    lithium_price = st.sidebar.slider("Lithium Price Change (%)", -30, 30, 0, 5)
    interest_rate = st.sidebar.slider("Interest Rate Change (pp)", -2.0, 2.0, 0.0, 0.25)
    incentives = st.sidebar.slider("Government Incentive Change (%)", -50, 50, 0, 5)
    carbon_tax = st.sidebar.slider("Carbon Tax Change (%)", -20, 50, 0, 5)
    gdp_growth = st.sidebar.slider("GDP Growth Change (%)", -5, 10, 0, 1)

    # Build adjustments
    adjustments = {
        "oil_price_change": oil_price,
        "lithium_price_change": lithium_price,
        "interest_rate_change": interest_rate,
        "incentive_change": incentives,
        "carbon_tax_change": carbon_tax,
        "gdp_growth_change": gdp_growth,
    }

    # Project
    with st.spinner("Running scenario..."):
        try:
            projected = project_scenario(df, adjustments)
        except Exception as e:
            st.error(f"Projection error: {e}")
            return

    base = df[df["year"] == base_year]

    # ── Global Impact KPIs ────────────────────────────────────────────────────
    total_base_sales = base["ev_sales"].sum()
    total_proj_sales = projected["ev_sales"].sum()
    delta_sales = total_proj_sales - total_base_sales
    delta_pct = (delta_sales / total_base_sales * 100) if total_base_sales != 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Base Year EV Sales", f"{total_base_sales:,.0f}")
    col2.metric("Projected EV Sales", f"{total_proj_sales:,.0f}", delta=f"{delta_sales:+,.0f}")
    col3.metric("Change (%)", f"{delta_pct:+.2f}%")

    # ── Country-level comparison ──────────────────────────────────────────────
    st.subheader("🌐 Top 10 Countries Comparison")
    top_countries = base.groupby("country")["ev_sales"].sum().nlargest(10).index
    comp_df = base[base["country"].isin(top_countries)][["country", "ev_sales"]].merge(
        projected[["country", "ev_sales"]], on="country", suffixes=("_base", "_proj")
    )
    comp_df = comp_df.melt(id_vars="country", var_name="Scenario", value_name="Sales")
    comp_df["Scenario"] = comp_df["Scenario"].replace({"ev_sales_base": "Base", "ev_sales_proj": "Projected"})
    fig = px.bar(
        comp_df, x="country", y="Sales", color="Scenario", barmode="group",
        title="Base vs Projected EV Sales (Top 10)",
        color_discrete_sequence=["#1f77b4", "#ff7f0e"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Adoption Curve (simulated) ────────────────────────────────────────────
    st.subheader("📈 Adoption Curve Simulation")
    # Simulate next 5 years with simple CAGR extension
    years = [base_year + i for i in range(0, 6)]
    sales_forecast = []
    for yr in years:
        if yr == base_year:
            sales_forecast.append(total_base_sales)
        else:
            # Apply the same factor for subsequent years (simplistic)
            factor = (1 + delta_pct/100)
            sales_forecast.append(sales_forecast[-1] * factor)
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=years, y=sales_forecast, mode='lines+markers', name='Projected', line=dict(color='green')))
    fig_line.add_trace(go.Scatter(x=[base_year], y=[total_base_sales], mode='markers', marker=dict(color='blue', size=12), name='Base Year'))
    fig_line.update_layout(title="Global EV Sales Forecast (5-year outlook)", xaxis_title="Year", yaxis_title="EV Sales")
    st.plotly_chart(fig_line, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    csv = projected.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Projected Data (CSV)", csv, "scenario_projection.csv", "text/csv")