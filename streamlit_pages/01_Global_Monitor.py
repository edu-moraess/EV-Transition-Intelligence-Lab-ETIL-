# streamlit_pages/01_Global_Monitor.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analytics.global_ranking import GlobalRanking


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "")
    mode    = df.attrs.get("vehicle_mode", "")
    st.title("🌍 Global Monitor")
    st.caption(f"IEA Global EV Outlook | Dataset: {dataset} | Modo: {mode}")
    st.markdown("---")

    latest_year = int(df["year"].max())
    prev_year   = latest_year - 1
    df_l = df[df["year"] == latest_year]
    df_p = df[df["year"] == prev_year]

    total_l = df_l["ev_sales"].sum()
    total_p = df_p["ev_sales"].sum() if not df_p.empty else 0
    yoy     = (total_l - total_p) / total_p * 100 if total_p > 0 else 0

    top_country = df_l.groupby("country")["ev_sales"].sum().idxmax()
    top_share   = df_l.groupby("country")["ev_sales"].sum().max() / total_l * 100
    avg_gr      = df_l["growth_rate"].median()

    # ── KPIs nativos Streamlit ────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Vendas EV {latest_year}", f"{total_l/1e6:.2f}M", f"{yoy:+.1f}% YoY")
    c2.metric("Países monitorados", str(df["country"].nunique()))
    c3.metric("Líder", top_country, f"{top_share:.1f}% share")
    c4.metric("Crescimento mediano YoY", f"{avg_gr:.1f}%" if not np.isnan(avg_gr) else "N/A")

    st.markdown("---")

    # ── Mapa coroplético ──────────────────────────────────────────────────
    st.subheader(f"🗺️ EV Stock Share por País — {latest_year}")
    # Usar stock share se disponível, senão vendas
    if "ev_stock_share" in df_l.columns and df_l["ev_stock_share"].notna().any():
        map_df   = df_l.groupby("country")["ev_stock_share"].mean().reset_index()
        map_col  = "ev_stock_share"
        map_lbl  = "Stock Share (%)"
        map_note = "Métrica: EV Stock Share (% da frota total)"
    else:
        map_df  = df_l.groupby("country")["ev_sales"].sum().reset_index()
        map_col = "ev_sales"
        map_lbl = "Vendas EV"
        map_note = "Métrica: Vendas EV (unidades)"
    st.caption(map_note)

    fig_map = px.choropleth(
        map_df,
        locations=map_df.columns[0],
        locationmode="country names",
        color=map_col,
        hover_name=map_df.columns[0],
        color_continuous_scale="Viridis",
        labels={map_col: map_lbl},
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        geo=dict(showframe=False, showcoastlines=True),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # ── Evolução temporal ─────────────────────────────────────────────────
    with col_l:
        st.subheader("📈 Evolução global de vendas")
        ts = df.groupby("year")["ev_sales"].sum().reset_index()
        fig_ts = px.bar(
            ts, x="year", y="ev_sales",
            labels={"ev_sales": "Vendas EV", "year": "Ano"},
            color_discrete_sequence=["#636EFA"],
        )
        fig_ts.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320, showlegend=False)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Top 10 países ─────────────────────────────────────────────────────
    with col_r:
        st.subheader(f"🏆 Top 10 países — {latest_year}")
        top10 = (
            df_l.groupby("country")["ev_sales"].sum()
            .nlargest(10)
            .sort_values()
            .reset_index()
        )
        fig_bar = px.bar(
            top10, x="ev_sales", y="country", orientation="h",
            labels={"ev_sales": "Vendas EV", "country": ""},
            color="ev_sales", color_continuous_scale="Viridis",
            text=top10["ev_sales"].map(lambda v: f"{v/1e3:,.0f}k"),
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            margin=dict(l=0, r=60, t=10, b=0), height=320,
            showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── EVQTI Top 10 ──────────────────────────────────────────────────────
    st.subheader("📊 EVQTI — Top 10 (early-stage momentum)")
    try:
        rdf = GlobalRanking(df).compute_ranking()
        st.dataframe(rdf.head(10).reset_index(drop=True), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Ranking indisponível: {e}")
