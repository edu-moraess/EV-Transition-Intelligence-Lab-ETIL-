# streamlit_pages/01_Global_Monitor.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analytics.global_ranking import GlobalRanking

# Paleta consistente
_PALETTE = px.colors.sequential.Plasma


def _kpi(label, value, delta=None, delta_label=""):
    delta_html = ""
    if delta is not None:
        cls = "delta-up" if delta >= 0 else "delta-down"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="metric-delta {cls}">{arrow} {abs(delta):.1f}% {delta_label}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


def render(df: pd.DataFrame):
    st.title("🌍 Global Monitor")
    st.markdown("Panorama mundial da transição para veículos elétricos — IEA Global EV Outlook 2026")
    st.markdown("---")

    latest_year = int(df["year"].max())
    prev_year   = latest_year - 1
    df_latest   = df[df["year"] == latest_year]
    df_prev     = df[df["year"] == prev_year]

    total_latest = df_latest["ev_sales"].sum()
    total_prev   = df_prev["ev_sales"].sum()
    yoy          = (total_latest - total_prev) / total_prev * 100 if total_prev > 0 else 0
    n_countries  = df["country"].nunique()
    top_country  = df_latest.groupby("country")["ev_sales"].sum().idxmax()
    top_share_row = df_latest.nlargest(1, "ev_sales")
    top_share    = top_share_row["ev_sales"].values[0] / total_latest * 100

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi(f"Vendas EV em {latest_year}", f"{total_latest/1e6:.2f} M unidades", yoy, "vs ano anterior")
    with c2:
        _kpi("Países monitorados", str(n_countries))
    with c3:
        _kpi("Líder de mercado", top_country, top_share - 100, "% share")
    with c4:
        avg_gr = df_latest["growth_rate"].median()
        _kpi("Crescimento mediano YoY", f"{avg_gr:.1f}%" if not np.isnan(avg_gr) else "N/A")

    st.markdown("---")

    # Mapa coroplético
    st.subheader(f"🗺️ Vendas de EV por País — {latest_year}")
    map_data = df_latest.groupby("country")["ev_sales"].sum().reset_index()
    fig_map = px.choropleth(
        map_data,
        locations="country",
        locationmode="country names",
        color="ev_sales",
        hover_name="country",
        color_continuous_scale="Plasma",
        labels={"ev_sales": "Vendas EV"},
    )
    fig_map.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        geo=dict(bgcolor="#0d1117", showframe=False, showcoastlines=True,
                 coastlinecolor="#30363d", landcolor="#161b22", showocean=True,
                 oceancolor="#0d1117"),
        coloraxis_colorbar=dict(title="Vendas", tickfont=dict(color="#8b949e")),
        margin=dict(l=0,r=0,t=0,b=0), height=420,
        font=dict(color="#e6edf3")
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # Série temporal agregada
    with col_left:
        st.subheader("📈 Evolução global de vendas")
        ts = df.groupby("year")["ev_sales"].sum().reset_index()
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Bar(
            x=ts["year"], y=ts["ev_sales"]/1e6,
            marker_color="#388bfd", name="Vendas EV (M)"
        ))
        fig_ts.update_layout(
            xaxis_title="Ano", yaxis_title="Milhões de unidades",
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"), height=320,
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d")
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # Top 10 países
    with col_right:
        st.subheader(f"🏆 Top 10 países — {latest_year}")
        top10 = map_data.nlargest(10, "ev_sales").sort_values("ev_sales")
        fig_bar = go.Figure(go.Bar(
            x=top10["ev_sales"]/1e3, y=top10["country"],
            orientation="h",
            marker_color="#3fb950",
            text=(top10["ev_sales"]/1e3).map(lambda v: f"{v:,.0f}k"),
            textposition="outside"
        ))
        fig_bar.update_layout(
            xaxis_title="Vendas (mil unidades)", yaxis_title="",
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"), height=320,
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Ranking EVQTI — Top 10")
    try:
        ranking = GlobalRanking(df)
        rdf = ranking.compute_ranking()
        rdf["evqti_score"] = rdf["evqti_score"].round(3)
        st.dataframe(
            rdf.head(10).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
    except Exception as e:
        st.warning(f"Ranking indisponível: {e}")
