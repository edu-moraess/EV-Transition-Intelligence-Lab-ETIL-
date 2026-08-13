# streamlit_pages/17_Anomaly_Detection.py
"""Anomaly Detection — Professional Edition."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.advanced_analytics import detect_anomalies
import numpy as np

def render(df: pd.DataFrame):
    st.title("🕵️ Anomaly Detection")
    st.markdown("Identify countries with unusual EV adoption patterns, infrastructure gaps, or extreme growth using multiple detectors.")

    if df.empty:
        st.warning("No data.")
        return

    with st.spinner("Running anomaly detection..."):
        try:
            df_anom = detect_anomalies(df)
        except Exception as e:
            st.error(f"Anomaly detection error: {e}")
            return

    latest = df_anom.sort_values("year").groupby("country").last().reset_index()
    anomalies = latest[latest["is_anomaly"] == True]
    normal = latest[latest["is_anomaly"] == False]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Countries", len(latest))
    col2.metric("🔴 Anomalies", len(anomalies))
    col3.metric("🟢 Normal", len(normal))

    # ── Anomaly Map ───────────────────────────────────────────────────────────
    st.subheader("🗺️ Anomaly Map")
    fig_map = px.choropleth(
        latest,
        locations="country",
        locationmode="country names",
        color="is_anomaly",
        color_discrete_map={True: "red", False: "lightgrey"},
        title="Detected Anomalous Countries",
        hover_name="country",
    )
    fig_map.update_layout(geo=dict(showframe=False, showcoastlines=True))
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Feature comparison scatter ────────────────────────────────────────────
    st.subheader("📊 Feature Comparison")
    feature_options = ["ev_sales", "ev_stock", "penetration", "growth_rate",
                       "charging_points_per_100k", "gdp_per_capita", "population"]
    valid_features = [f for f in feature_options if f in latest.columns]
    if len(valid_features) >= 2:
        x_axis = st.selectbox("X-axis", valid_features, index=0)
        y_axis = st.selectbox("Y-axis", valid_features, index=min(1, len(valid_features)-1))
        fig_scatter = px.scatter(
            latest, x=x_axis, y=y_axis,
            color="is_anomaly",
            color_discrete_map={True: "red", False: "blue"},
            hover_name="country",
            title=f"{x_axis} vs {y_axis} (Anomalies in red)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Anomaly details table ─────────────────────────────────────────────────
    st.subheader("🔍 Anomalous Countries Details")
    if not anomalies.empty:
        display_cols = ["country"] + valid_features[:4] + ["is_anomaly"]
        st.dataframe(anomalies[display_cols], use_container_width=True, hide_index=True)
        csv = anomalies.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Anomalies (CSV)", csv, "anomalies.csv", "text/csv")
    else:
        st.success("No anomalies detected!")

    # ── Explanation text ──────────────────────────────────────────────────────
    with st.expander("ℹ️ How anomalies are detected"):
        st.write("We combine Isolation Forest, Local Outlier Factor, and One-Class SVM. A country is flagged as anomalous if at least two of the three detectors agree. Features used: EV sales, stock, penetration, growth, infrastructure, GDP, population.")