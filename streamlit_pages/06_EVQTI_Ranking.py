# streamlit_pages/06_EVQTI_Ranking.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analytics.global_ranking import GlobalRanking

_BG="#161b22"; _GRID="#21262d"; _TEXT="#e6edf3"

def render(df: pd.DataFrame):
    st.title("🏆 EVQTI Ranking")
    st.markdown("**EV Quant Transition Index** — Índice proprietário PCA + z-score")

    # Filtro de países com dados válidos
    df = df.dropna(subset=['ev_sales', 'cagr', 'evqti_score' if 'evqti_score' in df.columns else None])

    with st.spinner("Calculando EVQTI..."):
        try:
            ranking = GlobalRanking(df)
            rdf = ranking.compute_ranking()
        except Exception as e:
            st.error(f"Erro no cálculo EVQTI: {e}")
            return

    rdf["evqti_score"] = rdf["evqti_score"].round(4)
    n_show = st.slider("Países a exibir", 10, len(rdf), min(30, len(rdf)))

    # Bar chart
    top_n = rdf.head(n_show).sort_values("evqti_score")
    fig = go.Figure(go.Bar(
        x=top_n["evqti_score"],
        y=top_n["country"],
        orientation="h",
        marker_color=px.colors.sequential.Plasma,
        text=top_n["evqti_score"].map(lambda v: f"{v:.3f}"),
        textposition="outside"
    ))
    fig.update_layout(
        title=f"EVQTI — Top {n_show} países ({df.attrs.get('vehicle_mode', 'Cars')})",
        xaxis_title="Score EVQTI", yaxis_title="",
        paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_TEXT),
        height=max(400, n_show*22)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Scatter corrigido
    st.subheader("🔍 EVQTI × CAGR × Vendas")
    df_latest = df[df["year"] == df["year"].max()]
    df_meta = df_latest.groupby("country").agg(
        cagr=("cagr", "mean"),
        ev_sales=("ev_sales", "sum"),
        growth_rate=("growth_rate", "mean")
    ).reset_index()

    scatter_df = rdf.merge(df_meta, on="country", how="left")

    # Tratamento de NaN para o scatter
    scatter_df = scatter_df.dropna(subset=["evqti_score", "cagr", "ev_sales"])
    scatter_df = scatter_df[scatter_df["ev_sales"] > 0]

    if len(scatter_df) > 0:
        fig_sc = px.scatter(
            scatter_df,
            x="evqti_score", 
            y="cagr",
            size="ev_sales",
            color="evqti_score",
            hover_name="country",
            size_max=60,
            color_continuous_scale="Plasma",
            labels={"evqti_score":"Score EVQTI","cagr":"CAGR (%)","ev_sales":"Vendas EV"}
        )
        fig_sc.update_layout(
            paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_TEXT),
            height=460
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.warning("Não há dados suficientes para o scatter plot após filtragem.")

    # Tabela
    st.subheader("📋 Tabela completa")
    full_df = rdf.merge(df_meta, on="country", how="left")
    full_df = full_df.round({"evqti_score":4, "cagr":1})
    st.dataframe(full_df.head(30), use_container_width=True, hide_index=True)