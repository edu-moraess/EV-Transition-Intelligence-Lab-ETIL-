# streamlit_pages/01_Global_Monitor.py
"""
Global Monitor — landing page of ETIL.

Design principles (ADNOC AI Lab reference):
  - Lead with a clear statement of what the data shows
  - Context before numbers — every metric has a one-line explanation
  - Visual hierarchy: hero KPIs → trend chart → top countries → EVQTI preview
  - No map (removed): choropleth adds complexity without insight at global level
  - Replace map with: regional breakdown bar + YoY growth heatmap by region
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from analytics.global_ranking import GlobalRanking
from visualization.theme import chart_config, base_layout, bar_color_signed, NAVY, NAVY_LIGHT, NAVY_PALE, RED_DARK, GREEN_MED, AMBER, SLATE


# ── Regional groupings ───────────────────────────────────────────────────────
_REGIONS = {
    "Europe":         ["Germany","France","United Kingdom","Norway","Netherlands",
                       "Sweden","Austria","Belgium","Denmark","Finland","Spain",
                       "Italy","Switzerland","Portugal","Greece","Poland","Czech Republic",
                       "Hungary","Romania","Bulgaria","Croatia","Estonia","Latvia",
                       "Lithuania","Slovakia","Slovenia","Ireland","Luxembourg","Malta"],
    "Asia Pacific":   ["China","Japan","South Korea","India","Australia","New Zealand",
                       "Thailand","Indonesia","Philippines","Viet Nam","Malaysia",
                       "Singapore","Cambodia","Lao Pdr"],
    "Americas":       ["United States","Canada","Brazil","Mexico","Colombia","Chile"],
    "Middle East & Africa": ["South Africa","Africa","United Arab Emirates","Jordan"],
    "Other":          [],
}

def _assign_region(country: str) -> str:
    for region, countries in _REGIONS.items():
        if country in countries:
            return region
    return "Other"


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "IEA 2026")
    mode    = df.attrs.get("vehicle_mode", "Cars")

    # ── Hero header ──────────────────────────────────────────────────────────
    st.markdown("## ⚡ EV Transition Intelligence Lab")
    st.markdown(
        "Tracking the global shift to electric vehicles across **60+ countries** — "
        "where it's happening, how fast, and what the data says about where it's going next."
    )
    st.caption(f"Source: IEA Global EV Outlook · Dataset: {dataset} · Vehicle mode: {mode}")
    st.markdown("---")

    # ── Data prep ────────────────────────────────────────────────────────────
    latest = int(df["year"].max())
    prev   = latest - 1
    df_l   = df[df["year"] == latest]
    df_p   = df[df["year"] == prev]

    total_l = df_l["ev_sales"].sum()
    total_p = df_p["ev_sales"].sum() if not df_p.empty else 0
    yoy     = (total_l - total_p) / total_p * 100 if total_p > 0 else 0
    top_c   = df_l.groupby("country")["ev_sales"].sum().idxmax()
    top_shr = df_l.groupby("country")["ev_sales"].sum().max() / total_l * 100
    n_ctry  = df["country"].nunique()

    # Avg stock share where available
    if "ev_stock_share" in df_l.columns and df_l["ev_stock_share"].notna().any():
        avg_share = df_l["ev_stock_share"].median()
        share_label = f"{avg_share:.1f}%"
    else:
        share_label = "N/A"

    # ── Hero KPIs ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"Global EV sales — {latest}",
        f"{total_l/1e6:.2f} million",
        f"{yoy:+.1f}% vs {prev}",
        help="Total passenger EV units sold globally in the latest year."
    )
    c2.metric(
        "Market leader",
        top_c,
        f"{top_shr:.1f}% of global sales",
        help="Country with the highest EV sales volume."
    )
    c3.metric(
        "Countries tracked",
        str(n_ctry),
        help="Countries with sufficient IEA data for analysis."
    )
    c4.metric(
        "Median EV stock share",
        share_label,
        help="Share of EVs in the total vehicle fleet — median across all countries."
    )

    st.markdown("---")

    # ── Global sales trend ───────────────────────────────────────────────────
    st.markdown("### How fast is the world adopting EVs?")
    st.markdown(
        "Global EV sales have grown from near-zero in 2010 to tens of millions today. "
        "The chart below shows annual sales volume and year-on-year growth rate."
    )

    ts = df.groupby("year")["ev_sales"].sum().reset_index()
    ts["yoy"] = ts["ev_sales"].pct_change() * 100

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Bar(
        x=ts["year"], y=ts["ev_sales"] / 1e6,
        name="EV Sales (M units)",
        marker_color=NAVY, opacity=0.85,
        hovertemplate="Year %{x}: %{y:.2f}M units<extra></extra>",
    ))
    # YoY on secondary axis
    fig_ts.add_trace(go.Scatter(
        x=ts["year"], y=ts["yoy"],
        name="YoY growth (%)",
        mode="lines+markers",
        line=dict(color=AMBER, width=2),
        marker=dict(size=6, color=AMBER),
        yaxis="y2",
        hovertemplate="Year %{x}: %{y:.1f}% YoY<extra></extra>",
    ))
    lay_ts = base_layout(360)
    lay_ts.update({
        "yaxis": dict(title="Million units",
                      gridcolor="rgba(128,128,128,0.12)", tickfont=dict(size=10)),
        "yaxis2": dict(title="YoY growth (%)", overlaying="y", side="right",
                       showgrid=False, tickfont=dict(size=10), zeroline=False),
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                       bgcolor="rgba(0,0,0,0)"),
        "barmode": "overlay",
    })
    fig_ts.update_layout(**lay_ts)
    st.plotly_chart(fig_ts, use_container_width=True, config=chart_config())

    st.markdown("---")

    # ── Regional breakdown + Top 10 ──────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Where is growth happening?")
        st.caption("EV sales by world region — latest year")
        df_l2 = df_l.copy()
        df_l2["region"] = df_l2["country"].apply(_assign_region)
        regional = df_l2.groupby("region")["ev_sales"].sum().sort_values(ascending=True)
        regional = regional[regional > 0]

        fig_reg = go.Figure(go.Bar(
            x=regional.values / 1e6,
            y=regional.index,
            orientation="h",
            marker_color=[NAVY if r != "Other" else SLATE for r in regional.index],
            text=(regional.values / 1e6).round(2).astype(str) + "M",
            textposition="outside",
            textfont=dict(size=9),
            hovertemplate="%{y}: %{x:.2f}M units<extra></extra>",
        ))
        lay_r = base_layout(300)
        lay_r.update({"xaxis": {**lay_r.get("xaxis",{}), "title": "Million units"},
                       "yaxis": {**lay_r.get("yaxis",{}), "title": ""},
                       "showlegend": False,
                       "margin": {**lay_r.get("margin",{}), "r": 70}})
        fig_reg.update_layout(**lay_r)
        st.plotly_chart(fig_reg, use_container_width=True, config=chart_config())

    with col_r:
        st.markdown(f"### Top 10 fastest-growing markets")
        st.caption(f"YoY growth rate in {latest} — among countries with >1,000 EV sales")
        top_gr = (
            df_l[df_l["ev_sales"] > 1000]
            .sort_values("growth_rate", ascending=False)
            .head(10)[["country","ev_sales","growth_rate"]]
            .sort_values("growth_rate")
        )
        fig_gr = go.Figure(go.Bar(
            x=top_gr["growth_rate"],
            y=top_gr["country"],
            orientation="h",
            marker_color=GREEN_MED,
            text=top_gr["growth_rate"].map(lambda v: f"{v:+.0f}%"),
            textposition="outside",
            textfont=dict(size=9),
            hovertemplate="%{y}: %{x:+.1f}% YoY<extra></extra>",
        ))
        lay_g = base_layout(300)
        lay_g.update({"xaxis": {**lay_g.get("xaxis",{}), "title": "YoY growth (%)"},
                       "yaxis": {**lay_g.get("yaxis",{}), "title": ""},
                       "showlegend": False,
                       "margin": {**lay_g.get("margin",{}), "r": 70}})
        fig_gr.update_layout(**lay_g)
        st.plotly_chart(fig_gr, use_container_width=True, config=chart_config())

    st.markdown("---")

    # ── EV Stock Share leaders ───────────────────────────────────────────────
    if "ev_stock_share" in df_l.columns and df_l["ev_stock_share"].notna().any():
        st.markdown("### Which countries have gone furthest?")
        st.markdown(
            "EV stock share measures the percentage of all cars on the road that are electric. "
            "It tells a different story from sales volume — some smaller countries are far ahead."
        )
        share_top = (
            df_l[df_l["ev_stock_share"].notna()]
            .groupby("country")["ev_stock_share"]
            .mean()
            .nlargest(15)
            .sort_values()
            .reset_index()
        )
        norm = (share_top["ev_stock_share"] - share_top["ev_stock_share"].min()) / \
               (share_top["ev_stock_share"].max() - share_top["ev_stock_share"].min() + 1)
        fig_sh = go.Figure(go.Bar(
            x=share_top["ev_stock_share"],
            y=share_top["country"],
            orientation="h",
            marker=dict(
                color=norm,
                colorscale=[[0, NAVY_PALE], [1, NAVY]],
                showscale=False,
            ),
            text=share_top["ev_stock_share"].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(size=9),
            hovertemplate="%{y}: %{x:.1f}% EV stock share<extra></extra>",
        ))
        lay_sh = base_layout(360)
        lay_sh.update({"xaxis": {**lay_sh.get("xaxis",{}), "title": "EV stock share (%)"},
                        "yaxis": {**lay_sh.get("yaxis",{}), "title": ""},
                        "showlegend": False,
                        "margin": {**lay_sh.get("margin",{}), "r": 60}})
        fig_sh.update_layout(**lay_sh)
        st.plotly_chart(fig_sh, use_container_width=True, config=chart_config())

    st.markdown("---")

    # ── EVQTI preview ────────────────────────────────────────────────────────
    st.markdown("### Which emerging markets have the most momentum?")
    st.markdown(
        "The **EVQTI** (EV Quant Transition Index) identifies countries with low current EV "
        "penetration but strong growth signals — the markets most likely to accelerate next. "
        "High penetration countries like Norway score lower because their growth is naturally "
        "slowing as the market matures."
    )
    try:
        rdf = GlobalRanking(df).compute_ranking()
        rdf["evqti_score"] = rdf["evqti_score"].round(3)
        rdf.columns = ["Country", "Rank", "EVQTI Score"]
        st.dataframe(rdf.head(10).reset_index(drop=True),
                     use_container_width=True, hide_index=True)
        st.caption(
            "→ Go to **EVQTI Ranking** in the sidebar for the full ranking, "
            "IC validation and quintile analysis."
        )
    except Exception as e:
        st.warning(f"Ranking unavailable: {e}")
