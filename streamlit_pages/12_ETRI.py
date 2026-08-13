# streamlit_pages/12_ETRI.py
"""EV Transition Risk Index (ETRI) — Professional Edition."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.advanced_analytics import calculate_etri

def render(df: pd.DataFrame):
    st.title("⚠️ EV Transition Risk Index (ETRI)")
    st.markdown("Proprietary composite index measuring transition readiness (0 = critical risk, 100 = very low risk).")

    if df.empty:
        st.warning("No data available. Load an IEA dataset first.")
        return

    # ── Compute ETRI ──────────────────────────────────────────────────────────
    with st.spinner("Calculating ETRI..."):
        try:
            df_etri = calculate_etri(df)
        except Exception as e:
            st.error(f"ETRI calculation failed: {e}")
            return

    # Latest year per country
    latest = df_etri.sort_values("year").groupby("country").last().reset_index()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    avg_score = latest["ETRI"].mean()
    critical_count = (latest["ETRI_category"] == "Critical").sum()
    very_low_count = (latest["ETRI_category"] == "Very Low").sum()
    high_count = (latest["ETRI_category"] == "High").sum()

    with col1:
        st.metric("Global Average ETRI", f"{avg_score:.1f}")
    with col2:
        st.metric("Critical Risk", critical_count)
    with col3:
        st.metric("High Risk", high_count)
    with col4:
        st.metric("Very Low Risk", very_low_count)

    # ── Interactive World Map (year selection) ────────────────────────────────
    st.subheader("🌍 Global Risk Map")
    available_years = sorted(df_etri["year"].unique())
    if len(available_years) > 1:
        map_year = st.select_slider("Select Year", options=available_years, value=available_years[-1])
    else:
        map_year = available_years[0]

    map_df = df_etri[df_etri["year"] == map_year]
    fig_map = px.choropleth(
        map_df,
        locations="country",
        locationmode="country names",
        color="ETRI",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        title=f"ETRI by Country ({map_year})",
        hover_name="country",
        hover_data={"ETRI": ":.1f", "ETRI_category": True},
    )
    fig_map.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"))
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Distribution & Gauge ──────────────────────────────────────────────────
    col_dist, col_gauge = st.columns([2, 1])
    with col_dist:
        st.subheader("Score Distribution")
        fig_hist = px.histogram(
            latest, x="ETRI", nbins=20, marginal="box",
            title="Distribution of ETRI Scores",
            color_discrete_sequence=["#2ca02c"]
        )
        fig_hist.add_vline(x=avg_score, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_score:.1f}")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_gauge:
        st.subheader("Selected Country Gauge")
        country_list = sorted(latest["country"].unique())
        selected_country = st.selectbox("Choose a country", country_list, key="etri_gauge_country")
        country_row = latest[latest["country"] == selected_country]
        if not country_row.empty:
            etri_val = country_row["ETRI"].values[0]
            cat = country_row["ETRI_category"].values[0]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=etri_val,
                title={"text": f"{selected_country} ETRI"},
                delta={"reference": avg_score, "increasing": {"color": "green"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkgreen"},
                    "steps": [
                        {"range": [0, 20], "color": "red"},
                        {"range": [20, 40], "color": "orange"},
                        {"range": [40, 60], "color": "yellow"},
                        {"range": [60, 80], "color": "lightgreen"},
                        {"range": [80, 100], "color": "green"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.8, "value": etri_val},
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f"**Category:** {cat}")

    # ── Ranking Table ─────────────────────────────────────────────────────────
    st.subheader("🏆 Country Ranking")
    ranking = latest[["country", "ETRI", "ETRI_category"]].sort_values("ETRI", ascending=False)
    st.dataframe(ranking, use_container_width=True, hide_index=True)

    # Downloads
    csv = ranking.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "etri_ranking.csv", "text/csv")