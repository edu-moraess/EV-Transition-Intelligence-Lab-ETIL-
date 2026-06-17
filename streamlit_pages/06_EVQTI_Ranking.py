# streamlit_pages/06_EVQTI_Ranking.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from analytics.global_ranking import GlobalRanking

_BG="#161b22"; _GRID="#21262d"; _TEXT="#e6edf3"


def render(df: pd.DataFrame):
    st.title("🏆 EVQTI Ranking")
    st.markdown("**EV Quant Transition Index** — Índice proprietário PCA + z-score + ponderação por entropia")
    st.markdown("---")

    with st.spinner("Calculando EVQTI..."):
        try:
            ranking = GlobalRanking(df)
            rdf = ranking.compute_ranking()
        except Exception as e:
            st.error(f"Erro no cálculo EVQTI: {e}"); return

    rdf["evqti_score"] = rdf["evqti_score"].round(4)
    n_show = st.slider("Países a exibir", 10, len(rdf), min(30, len(rdf)))

    # Bar chart horizontal
    top_n = rdf.head(n_show).sort_values("evqti_score")
    colors = px.colors.sequential.Plasma
    n_col = len(colors)
    bar_colors = [colors[int(i / len(top_n) * (n_col-1))] for i in range(len(top_n))]

    fig = go.Figure(go.Bar(
        x=top_n["evqti_score"],
        y=top_n["country"],
        orientation="h",
        marker_color=bar_colors,
        text=top_n["evqti_score"].map(lambda v: f"{v:.3f}"),
        textposition="outside"
    ))
    fig.update_layout(
        title=f"EVQTI — Top {n_show} países",
        xaxis_title="Score EVQTI", yaxis_title="",
        paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_TEXT),
        height=max(400, n_show*22), margin=dict(l=0,r=80,t=40,b=0),
        xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Scatter: EVQTI vs CAGR
    st.subheader("🔍 EVQTI × CAGR × Vendas")
    df_latest = df[df["year"] == df["year"].max()]
    df_meta   = df_latest.groupby("country").agg(
        cagr=("cagr","mean"),
        ev_sales=("ev_sales","sum"),
        growth_rate=("growth_rate","mean")
    ).reset_index()
    scatter_df = rdf.merge(df_meta, on="country", how="left")

    fig_sc = px.scatter(
        scatter_df,
        x="evqti_score", y="cagr",
        size="ev_sales", color="evqti_score",
        hover_name="country",
        color_continuous_scale="Plasma",
        size_max=50,
        labels={"evqti_score":"Score EVQTI","cagr":"CAGR (%)","ev_sales":"Vendas EV"}
    )
    fig_sc.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_TEXT),
        height=460, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(gridcolor=_GRID, title="Score EVQTI"),
        yaxis=dict(gridcolor=_GRID, title="CAGR (%)"),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # Tabela completa
    st.subheader("📋 Tabela completa")
    full_df = rdf.merge(df_meta[["country","cagr","ev_sales","growth_rate"]], on="country", how="left")
    full_df["cagr"]        = full_df["cagr"].round(1)
    full_df["ev_sales"]    = full_df["ev_sales"].round(0).astype("Int64")
    full_df["growth_rate"] = full_df["growth_rate"].round(1)
    full_df.columns        = ["País","Rank","Score EVQTI","CAGR (%)","Vendas EV (último ano)","Crescimento YoY (%)"]

    st.dataframe(full_df, use_container_width=True, hide_index=True)

    # Metodologia
    with st.expander("ℹ️ Metodologia EVQTI"):
        st.markdown("""
**Features utilizadas:**
- `penetration` / `ev_stock_share` — participação de mercado EV
- `growth_rate` — variação YoY de vendas
- `cagr` — taxa de crescimento anual composta
- `momentum_3y` — momentum de 3 anos
- `ev_per_capita` — vendas per capita (quando disponível)
- `ev_per_charger` — EVs por ponto de recarga

**Pipeline:**
1. Média histórica por país em cada feature
2. Imputação por mediana para NaNs
3. Padronização Z-score (`StandardScaler`)
4. PCA — extração do primeiro componente principal como score composto
5. Reorientação: score mais alto = transição mais avançada

**Fonte:** IEA Global EV Outlook 2026
        """)
