# streamlit_pages/06_EVQTI_Ranking.py
"""
Página: EVQTI Ranking - ranking quantitativo global.
"""
import streamlit as st
import pandas as pd
from analytics.global_ranking import GlobalRanking
from visualization.charts import Charts

def render(df_processed):
    st.title("🏆 EVQTI Ranking")
    st.markdown("Índice Quantitativo de Transição de Veículos Elétricos")

    ranking = GlobalRanking(df_processed)
    ranking_df = ranking.compute_ranking()
    st.dataframe(ranking_df)

    # Gráfico de barras
    fig = Charts.bar_chart(ranking_df.head(20), x="country", y="evqti_score",
                           title="Top 20 EVQTI")
    st.plotly_chart(fig, use_container_width=True)