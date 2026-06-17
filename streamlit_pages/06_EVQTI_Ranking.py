# streamlit_pages/06_EVQTI_Ranking.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from analytics.global_ranking import GlobalRanking

_BG = "#161b22"
_GRID = "#21262d"
_TEXT = "#e6edf3"


def render(df: pd.DataFrame):
    st.title("🏆 EVQTI Ranking")
    st.markdown("**EV Quant Transition Index** — Índice proprietário PCA + z-score + ponderação por entropia")
    st.markdown("---")

    # =========================
    # EVQTI CALC
    # =========================
    with st.spinner("Calculando EVQTI..."):
        try:
            ranking = GlobalRanking(df)
            rdf = ranking.compute_ranking()
        except Exception as e:
            st.error(f"Erro no cálculo EVQTI: {e}")
            return

    rdf["evqti_score"] = rdf["evqti_score"].round(4)

    n_show = st.slider(
        "Países a exibir",
        10,
        len(rdf),
        min(30, len(rdf))
    )

    # =========================
    # BAR CHART
    # =========================
    top_n = rdf.head(n_show).sort_values("evqti_score")

    colors = px.colors.sequential.Plasma
    n_col = len(colors)
    bar_colors = [
        colors[int(i / len(top_n) * (n_col - 1))]
        for i in range(len(top_n))
    ]

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
        xaxis_title="Score EVQTI",
        yaxis_title="",
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT),
        height=max(400, n_show * 22),
        margin=dict(l=0, r=80, t=40, b=0),
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(gridcolor=_GRID)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # =========================
    # DATA PREP (FIXED)
    # =========================
    st.subheader("🔍 EVQTI × CAGR × Vendas")

    df_latest = df[df["year"] == df["year"].max()]

    df_meta = df_latest.groupby("country").agg(
        cagr=("cagr", "mean"),
        ev_sales=("ev_sales", "sum"),
        growth_rate=("growth_rate", "mean")
    ).reset_index()

    # 🔥 merge mantendo TODOS países do ranking
    scatter_df = rdf.merge(df_meta, on="country", how="left")

    # =========================
    # NA SAFE LAYER (CRITICAL FIX)
    # =========================
    scatter_df["ev_sales"] = scatter_df["ev_sales"].fillna(0)
    scatter_df["cagr"] = scatter_df["cagr"].fillna(scatter_df["cagr"].median())
    scatter_df["growth_rate"] = scatter_df["growth_rate"].fillna(0)

    # =========================
    # QUANT SCALING (ROBUST SIZE)
    # =========================
    scatter_df["ev_sales_log"] = np.log1p(scatter_df["ev_sales"])

    # =========================
    # SCATTER PLOT
    # =========================
    fig_sc = px.scatter(
        scatter_df,
        x="evqti_score",
        y="cagr",
        size="ev_sales_log",
        color="evqti_score",
        hover_name="country",
        color_continuous_scale="Plasma",
        size_max=50,
        labels={
            "evqti_score": "Score EVQTI",
            "cagr": "CAGR (%)",
            "ev_sales_log": "Vendas EV (log)"
        }
    )

    fig_sc.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT),
        height=460,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor=_GRID, title="Score EVQTI"),
        yaxis=dict(gridcolor=_GRID, title="CAGR (%)"),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # =========================
    # TABLE
    # =========================
    st.subheader("📋 Tabela completa")

    full_df = rdf.merge(
        df_meta[["country", "cagr", "ev_sales", "growth_rate"]],
        on="country",
        how="left"
    )

    full_df["cagr"] = full_df["cagr"].fillna(0).round(1)
    full_df["ev_sales"] = full_df["ev_sales"].fillna(0).round(0).astype("Int64")
    full_df["growth_rate"] = full_df["growth_rate"].fillna(0).round(1)

    full_df.columns = [
        "País",
        "Rank",
        "Score EVQTI",
        "CAGR (%)",
        "Vendas EV (último ano)",
        "Crescimento YoY (%)"
    ]

    st.dataframe(full_df, use_container_width=True, hide_index=True)

    # =========================
    # METHODOLOGY
    # =========================
    with st.expander("ℹ️ Metodologia EVQTI"):
        st.markdown("""
**Features utilizadas:**
- `penetration` / `ev_stock_share`
- `growth_rate`
- `cagr`
- `momentum_3y`
- `ev_per_capita`
- `ev_per_charger`

**Pipeline:**
1. Média histórica por país
2. Imputação por mediana
3. Z-score normalization
4. PCA (1º componente)
5. Score final reorientado (maior = melhor transição)

**Fonte:** IEA Global EV Outlook 2026
        """)