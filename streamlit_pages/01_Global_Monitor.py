# streamlit_pages/01_Global_Monitor.py
"""
Página: Global Monitor - visão geral mundial.
"""
import streamlit as st
import pandas as pd
from visualization.world_map import WorldMap
from analytics.global_ranking import GlobalRanking

def render(df_processed):
    st.title("🌍 Global Monitor")
    st.markdown("Panorama mundial da transição elétrica")

    # Métricas gerais
    total_ev = df_processed["ev_sales"].sum()
    total_countries = df_processed["country"].nunique()
    latest_year = df_processed["year"].max()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de EVs (ano mais recente)", f"{total_ev:,.0f}")
    col2.metric("Países analisados", total_countries)
    col3.metric("Ano atual", latest_year)

    # Mapa
    latest_data = df_processed[df_processed["year"] == latest_year].groupby("country")["ev_sales"].sum().reset_index()
    fig_map = WorldMap.choropleth(latest_data, value_col="ev_sales", title=f"Vendas de EV em {latest_year}")
    st.plotly_chart(fig_map, use_container_width=True)

    # Ranking
    ranking = GlobalRanking(df_processed)
    ranking_df = ranking.compute_ranking()
    st.subheader("🏆 Ranking EVQTI")
    st.dataframe(ranking_df.head(10))