# streamlit_pages/12_ETRI.py
"""EV Transition Risk Index (ETRI) page."""
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.advanced_analytics import calculate_etri

def render(df: pd.DataFrame):
    st.title("⚠️ EV Transition Risk Index (ETRI)")
    st.markdown("Proprietary index measuring transition readiness and risk (0 = critical risk, 100 = very low risk).")

    if df.empty:
        st.warning("No data available. Load an IEA dataset first.")
        return

    try:
        df_etri = calculate_etri(df)
    except Exception as e:
        st.error(f"ETRI calculation failed: {e}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average ETRI", f"{df_etri['ETRI'].mean():.1f}")
    with col2:
        critical = (df_etri['ETRI_category'] == 'Critical').sum()
        st.metric("Countries in Critical", critical)
    with col3:
        very_low = (df_etri['ETRI_category'] == 'Very Low').sum()
        st.metric("Very Low Risk", very_low)

    st.subheader("ETRI Distribution")
    fig = px.histogram(df_etri.dropna(subset=['ETRI']), x='ETRI', nbins=20, marginal='box',
                       title='Distribution of ETRI Scores')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Country Ranking")
    ranking = df_etri.groupby('country').last().reset_index()[['country', 'ETRI', 'ETRI_category']]
    ranking = ranking.sort_values('ETRI', ascending=False)
    st.dataframe(ranking, use_container_width=True)

    csv = ranking.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "etri_ranking.csv", "text/csv")

    st.subheader("Global Risk Map")
    fig_map = px.choropleth(ranking, locations='country', locationmode='country names',
                            color='ETRI', color_continuous_scale='RdYlGn', range_color=[0, 100],
                            title='ETRI by Country')
    st.plotly_chart(fig_map, use_container_width=True)