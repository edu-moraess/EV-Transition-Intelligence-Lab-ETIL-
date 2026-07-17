# streamlit_pages/17_Anomaly_Detection.py
"""Anomaly Detection page."""
import streamlit as st
import plotly.express as px
import pandas as pd
from utils.advanced_analytics import detect_anomalies

def render(df: pd.DataFrame):
    st.title("🕵️ Anomaly Detection")
    st.markdown("Detect unusual adoption patterns, infrastructure gaps, or extreme growth.")

    if df.empty:
        st.warning("No data.")
        return

    try:
        df_anom = detect_anomalies(df)
    except Exception as e:
        st.error(f"Anomaly detection error: {e}")
        return

    latest = df_anom.sort_values("year").groupby("country").last().reset_index()
    anomalies = latest[latest["is_anomaly"] == True]
    normal = latest[latest["is_anomaly"] == False]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Countries", len(latest))
    col2.metric("Anomalies Detected", len(anomalies))
    col3.metric("Normal", len(normal))

    st.subheader("Anomaly Map")
    fig = px.scatter_geo(latest, locations="country", locationmode="country names",
                         color="is_anomaly", color_discrete_map={True: "red", False: "blue"},
                         hover_name="country", title="Anomalous Countries")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomalous Countries Details")
    st.dataframe(anomalies[["country", "ev_sales", "penetration", "growth_rate"]])

    csv = anomalies.to_csv(index=False).encode('utf-8')
    st.download_button("Download Anomalies CSV", csv, "anomalies.csv", "text/csv")