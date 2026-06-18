# streamlit_pages/01_Global_Monitor.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from analytics.global_ranking import GlobalRanking
from visualization.theme import (
    base_layout, NAVY, NAVY_LIGHT, NAVY_PALE,
    RED_DARK, GREEN_MED, SLATE, BORDER, SEQ_MAP
)


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "")
    mode    = df.attrs.get("vehicle_mode", "")
    st.title("🌍 Global Monitor")
    st.caption(f"IEA Global EV Outlook | Dataset: {dataset} | Modo: {mode}")
    st.markdown("---")

    latest = int(df["year"].max())
    prev   = latest - 1
    df_l   = df[df["year"] == latest]
    df_p   = df[df["year"] == prev]

    total_l = df_l["ev_sales"].sum()
    total_p = df_p["ev_sales"].sum() if not df_p.empty else 0
    yoy     = (total_l - total_p) / total_p * 100 if total_p > 0 else 0
    top_c   = df_l.groupby("country")["ev_sales"].sum().idxmax()
    top_shr = df_l.groupby("country")["ev_sales"].sum().max() / total_l * 100
    med_gr  = df_l["growth_rate"].median()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Vendas EV {latest}", f"{total_l/1e6:.2f}M", f"{yoy:+.1f}% YoY")
    c2.metric("Países monitorados", str(df["country"].nunique()))
    c3.metric("Líder de mercado", top_c, f"{top_shr:.1f}% share")
    c4.metric("Crescimento mediano YoY", f"{med_gr:.1f}%" if not np.isnan(med_gr) else "N/A")

    st.markdown("---")

    # ── Mapa ──────────────────────────────────────────────────────────────
    st.subheader(f"🗺️ EV Stock Share por País — {latest}")
    use_share = "ev_stock_share" in df_l.columns and df_l["ev_stock_share"].notna().any()
    if use_share:
        map_df  = df_l.groupby("country")["ev_stock_share"].mean().reset_index()
        map_col = "ev_stock_share"; map_lbl = "EV Stock Share (%)"
        st.caption("Métrica: EV Stock Share (% da frota total) — mais informativo que vendas brutas")
    else:
        map_df  = df_l.groupby("country")["ev_sales"].sum().reset_index()
        map_col = "ev_sales"; map_lbl = "Vendas EV"
        st.caption("Métrica: Vendas EV (unidades)")

    fig_map = go.Figure(go.Choropleth(
        locations=map_df["country"],
        locationmode="country names",
        z=map_df[map_col],
        colorscale=SEQ_MAP,
        colorbar=dict(title=map_lbl, thickness=12, len=0.7,
                      tickfont=dict(size=10)),
        hovertemplate="<b>%{location}</b><br>" + map_lbl + ": %{z:.2f}<extra></extra>",
    ))
    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=True,
                 coastlinecolor=BORDER, showland=True,
                 landcolor="#f4f6f9", showocean=True,
                 oceancolor="#eef2f7", showlakes=False),
        margin=dict(l=0,r=0,t=0,b=0), height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # ── Série temporal ────────────────────────────────────────────────────
    with col_l:
        st.markdown("##### Evolução global de vendas EV")
        ts = df.groupby("year")["ev_sales"].sum().reset_index()
        fig_ts = go.Figure(go.Bar(
            x=ts["year"], y=ts["ev_sales"]/1e6,
            marker_color=NAVY,
            hovertemplate="Ano: %{x}<br>Vendas: %{y:.2f}M<extra></extra>",
        ))
        lay = base_layout(height=300)
        lay.update({"yaxis": {**lay.get("yaxis",{}), "title": "Milhões de unidades"},
                     "showlegend": False})
        fig_ts.update_layout(**lay)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Top 10 ────────────────────────────────────────────────────────────
    with col_r:
        st.markdown(f"##### Top 10 países — {latest}")
        top10 = (
            df_l.groupby("country")["ev_sales"].sum()
            .nlargest(10).sort_values().reset_index()
        )
        norm = (top10["ev_sales"] - top10["ev_sales"].min()) / \
               (top10["ev_sales"].max() - top10["ev_sales"].min() + 1)
        fig_bar = go.Figure(go.Bar(
            x=top10["ev_sales"]/1e3, y=top10["country"],
            orientation="h",
            marker=dict(
                color=norm,
                colorscale=[[0,"#c8d4e8"],[1,NAVY]],
                showscale=False,
            ),
            text=(top10["ev_sales"]/1e3).map(lambda v: f"{v:,.0f}k"),
            textposition="outside",
            textfont=dict(size=9),
            hovertemplate="%{y}: %{x:,.0f}k unidades<extra></extra>",
        ))
        lay2 = base_layout(height=300)
        lay2.update({"xaxis": {**lay2.get("xaxis",{}), "title": "k unidades"},
                      "yaxis": {**lay2.get("yaxis",{}), "title": ""},
                      "margin": {**lay2.get("margin",{}), "r": 70},
                      "showlegend": False})
        fig_bar.update_layout(**lay2)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── EVQTI preview ─────────────────────────────────────────────────────
    st.subheader("📊 EVQTI — Top 10 por early-stage momentum")
    try:
        rdf = GlobalRanking(df).compute_ranking()
        rdf["evqti_score"] = rdf["evqti_score"].round(4)
        st.dataframe(rdf.head(10).reset_index(drop=True),
                     use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Ranking indisponível: {e}")
