# streamlit_pages/01_Global_Monitor.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from analytics.global_ranking import GlobalRanking
from visualization.theme import base_layout, NAVY, SLATE, BORDER, SEQ_MAP


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "")
    mode    = df.attrs.get("vehicle_mode", "")
    st.title("Global Monitor")
    st.caption(f"IEA Global EV Outlook · Dataset: {dataset} · Mode: {mode}")
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
    c1.metric(f"EV Sales {latest}",     f"{total_l/1e6:.2f}M units",
              f"{yoy:+.1f}% YoY")
    c2.metric("Countries monitored",    str(df["country"].nunique()))
    c3.metric("Market leader",          top_c, f"{top_shr:.1f}% share")
    c4.metric("Median YoY growth",      f"{med_gr:.1f}%" if not np.isnan(med_gr) else "N/A")

    st.markdown("---")

    # ── Choropleth map ────────────────────────────────────────────────────
    use_share = "ev_stock_share" in df_l.columns and df_l["ev_stock_share"].notna().any()
    if use_share:
        map_df  = df_l.groupby("country")["ev_stock_share"].mean().reset_index()
        map_col = "ev_stock_share"
        map_lbl = "EV Stock Share (%)"
        st.subheader(f"EV Stock Share by Country — {latest}")
        st.caption("Share of EVs in total vehicle stock — more informative than raw sales volume")
    else:
        map_df  = df_l.groupby("country")["ev_sales"].sum().reset_index()
        map_col = "ev_sales"
        map_lbl = "EV Sales"
        st.subheader(f"EV Sales by Country — {latest}")

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
                 coastlinecolor="rgba(128,128,128,0.3)",
                 showland=True, landcolor="rgba(240,243,248,1)",
                 showocean=True, oceancolor="rgba(230,236,245,1)"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # ── Time series ───────────────────────────────────────────────────────
    with col_l:
        st.markdown("##### Global EV Sales — Historical")
        ts = df.groupby("year")["ev_sales"].sum().reset_index()
        fig_ts = go.Figure(go.Bar(
            x=ts["year"], y=ts["ev_sales"] / 1e6,
            marker_color=NAVY,
            hovertemplate="Year %{x}: %{y:.2f}M units<extra></extra>",
        ))
        lay = base_layout(300)
        lay.update({"yaxis": {**lay.get("yaxis", {}), "title": "Million units"},
                     "showlegend": False})
        fig_ts.update_layout(**lay)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Top 10 ────────────────────────────────────────────────────────────
    with col_r:
        st.markdown(f"##### Top 10 Countries — {latest}")
        top10 = (
            df_l.groupby("country")["ev_sales"].sum()
            .nlargest(10).sort_values().reset_index()
        )
        norm = (top10["ev_sales"] - top10["ev_sales"].min()) / \
               (top10["ev_sales"].max() - top10["ev_sales"].min() + 1)
        fig_bar = go.Figure(go.Bar(
            x=top10["ev_sales"] / 1e3, y=top10["country"],
            orientation="h",
            marker=dict(color=norm,
                        colorscale=[[0, "#c8d4e8"], [1, NAVY]],
                        showscale=False),
            text=(top10["ev_sales"] / 1e3).map(lambda v: f"{v:,.0f}k"),
            textposition="outside",
            textfont=dict(size=9),
            hovertemplate="%{y}: %{x:,.0f}k units<extra></extra>",
        ))
        lay2 = base_layout(300)
        lay2.update({"xaxis": {**lay2.get("xaxis", {}), "title": "Thousand units"},
                      "yaxis": {**lay2.get("yaxis", {}), "title": ""},
                      "margin": {**lay2.get("margin", {}), "r": 70},
                      "showlegend": False})
        fig_bar.update_layout(**lay2)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── EVQTI preview ─────────────────────────────────────────────────────
    st.subheader("EVQTI — Top 10 by Early-Stage Momentum")
    try:
        rdf = GlobalRanking(df).compute_ranking()
        rdf["evqti_score"] = rdf["evqti_score"].round(4)
        st.dataframe(rdf.head(10).reset_index(drop=True),
                     use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Ranking unavailable: {e}")
