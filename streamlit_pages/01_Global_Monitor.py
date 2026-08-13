# streamlit_pages/01_Global_Monitor.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analytics.global_ranking import GlobalRanking
from visualization.theme import (
    chart_config, base_layout, NAVY, NAVY_LIGHT, NAVY_PALE, GREEN_MED, AMBER, GOLD, SLATE, SEQ_MAP,
)

_REGIONS = {
    "Europe":              ["Germany","France","United Kingdom","Norway","Netherlands",
                            "Sweden","Austria","Belgium","Denmark","Finland","Spain",
                            "Italy","Switzerland","Portugal","Greece","Poland","Czech Republic",
                            "Hungary","Romania","Bulgaria","Croatia","Estonia","Latvia",
                            "Lithuania","Slovakia","Slovenia","Ireland","Luxembourg","Malta"],
    "Asia Pacific":        ["China","Japan","South Korea","India","Australia","New Zealand",
                            "Thailand","Indonesia","Philippines","Viet Nam","Malaysia",
                            "Singapore","Cambodia","Lao Pdr"],
    "Americas":            ["United States","Canada","Brazil","Mexico","Colombia","Chile"],
    "Middle East & Africa":["South Africa","Africa","United Arab Emirates","Jordan"],
}

def _region(c):
    for r, cs in _REGIONS.items():
        if c in cs: return r
    return "Other"


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "IEA 2026")
    mode    = df.attrs.get("vehicle_mode", "Cars")

    st.markdown("## Global EV transition monitor")
    st.markdown(
        "A cross-country view of observed EV adoption, market scale and transition momentum. "
        "All reported levels are drawn from the selected IEA historical panel."
    )
    st.caption(f"Source: IEA Global EV Outlook · Release: {dataset} · Universe: {mode} · Last observed year shown below")
    st.markdown("---")

    latest = int(df["year"].max())
    prev   = latest - 1
    df_l   = df[df["year"] == latest]
    df_p   = df[df["year"] == prev]

    total_l    = df_l["ev_sales"].sum()
    total_p    = df_p["ev_sales"].sum() if not df_p.empty else 0
    yoy        = (total_l - total_p) / total_p * 100 if total_p > 0 else 0
    top_c      = df_l.groupby("country")["ev_sales"].sum().idxmax()
    top_shr    = df_l.groupby("country")["ev_sales"].sum().max() / total_l * 100
    n_ctry     = df["country"].nunique()
    median_shr = df_l["ev_stock_share"].median() if "ev_stock_share" in df_l.columns else None

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Global EV sales — {latest}", f"{total_l/1e6:.2f}M",
              f"{yoy:+.1f}% vs {prev}",
              help="Total EV units sold globally in the latest year.")
    c2.metric("Market leader", top_c, f"{top_shr:.1f}% of global sales")
    c3.metric("Countries tracked", str(n_ctry))
    c4.metric("Median EV stock share",
              f"{median_shr:.1f}%" if median_shr else "N/A",
              help="Share of EVs in the total vehicle fleet — median across all countries.")

    st.markdown("---")

    metric_map = {
        "EV sales share (%)": "ev_sales_share",
        "EV stock share (%)": "ev_stock_share",
        "Annual EV sales (units)": "ev_sales",
        "Public charging points": "charging_points",
    }
    available_metrics = {label: column for label, column in metric_map.items() if column in df_l.columns and df_l[column].notna().any()}
    if available_metrics:
        map_col, map_note = st.columns([1, 3])
        with map_col:
            map_label = st.selectbox("Map metric", list(available_metrics), label_visibility="collapsed")
        with map_note:
            st.caption(f"Observed cross-section for {latest}. Countries without a reported value are intentionally omitted.")
        map_metric = available_metrics[map_label]
        map_df = df_l[["country", map_metric]].dropna()
        if not map_df.empty:
            fig_map = px.choropleth(
                map_df,
                locations="country",
                locationmode="country names",
                color=map_metric,
                hover_name="country",
                color_continuous_scale=SEQ_MAP,
                labels={map_metric: map_label},
            )
            fig_map.update_geos(showframe=False, showcoastlines=True, coastlinecolor="rgba(70,88,106,0.32)", projection_type="natural earth")
            fig_map.update_layout(**base_layout(360, f"Geographic distribution — {map_label}"), coloraxis_colorbar={"thickness": 12, "len": 0.68})
            st.plotly_chart(fig_map, width="stretch", config=chart_config())
            st.markdown("---")

    # ── Toggle: Volume vs Per Capita ─────────────────────────────────────────
    st.markdown("### Global sales trend")
    has_population = "population" in df.columns and df["population"].notna().any()
    views = ["Volume", "Per Capita"] if has_population else ["Volume"]
    view_mode = st.radio(
        "View",
        views,
        horizontal=True,
        help="Per capita is available only when the selected panel contains a population series.",
    )

    aggregations = {"ev_sales": ("ev_sales", "sum")}
    if has_population:
        aggregations["population"] = ("population", "sum")
    ts = df.groupby("year").agg(**aggregations).reset_index()
    ts["yoy"] = ts["ev_sales"].pct_change() * 100
    ts["per_capita"] = np.nan
    if has_population:
        ts["per_capita"] = np.where(
            ts["population"] > 1e6,
            ts["ev_sales"] / ts["population"] * 1000,
            np.nan,
        )

    y_col   = "per_capita" if view_mode == "Per Capita" else "ev_sales"
    y_label = "EVs per 1,000 inhabitants" if view_mode == "Per Capita" else "Million units"
    y_vals  = ts[y_col] / (1 if view_mode == "Per Capita" else 1e6)

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Bar(
        x=ts["year"], y=y_vals, name=y_label,
        marker_color=NAVY, opacity=0.85,
        hovertemplate=f"Year %{{x}}: %{{y:.{'2f' if view_mode=='Per Capita' else '2f'}}} {y_label}<extra></extra>",
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts["year"], y=ts["yoy"], name="YoY growth (%)",
        mode="lines+markers", line=dict(color=GOLD, width=2.5),
        marker=dict(size=7, color=GOLD), yaxis="y2",
        hovertemplate="Year %{x}: %{y:.1f}% YoY<extra></extra>",
    ))
    lay_ts = base_layout(360)
    lay_ts.update({
        "yaxis":  dict(title=y_label, gridcolor="rgba(128,128,128,0.12)"),
        "yaxis2": dict(title="YoY growth (%)", overlaying="y", side="right",
                       showgrid=False, zeroline=False),
        "legend": dict(orientation="h", yanchor="bottom", y=1.02,
                       xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        "barmode": "overlay",
    })
    fig_ts.update_layout(**lay_ts)
    st.plotly_chart(fig_ts, width="stretch", config=chart_config())

    st.markdown("---")

    # ── Heatmap temporal: EV Sales Share por país × ano ───────────────────────
    if "ev_sales_share" in df.columns and df["ev_sales_share"].notna().any():
        st.markdown("### The adoption wave — EV sales share by country over time")
        st.markdown(
            "Each cell shows the percentage of **new cars sold** that were electric in that year. "
            "Watch the wave of adoption spread — first Scandinavia, then Europe, now reaching emerging markets."
        )
        hm_data = (
            df[df["ev_sales_share"].notna()]
            .groupby(["country","year"])["ev_sales_share"]
            .mean()
            .reset_index()
        )
        # Keep only countries with at least some share data and recent data
        recent_share = hm_data[hm_data["year"] >= latest - 2].groupby("country")["ev_sales_share"].max()
        top_countries = recent_share.nlargest(25).index.tolist()
        hm_top = hm_data[hm_data["country"].isin(top_countries)]
        hm_pivot = hm_top.pivot(index="country", columns="year", values="ev_sales_share").fillna(0)
        hm_pivot = hm_pivot.reindex(
            hm_pivot.iloc[:, -1].sort_values(ascending=False).index
        )

        fig_hm = go.Figure(go.Heatmap(
            z=hm_pivot.values,
            x=[str(y) for y in hm_pivot.columns],
            y=hm_pivot.index.tolist(),
            colorscale=[[0,"#f0f4ff"],[0.25,NAVY_PALE],[0.6,NAVY_LIGHT],[1.0,NAVY]],
            text=np.where(hm_pivot.values > 0,
                          np.vectorize(lambda v: f"{v:.1f}%")(hm_pivot.values), ""),
            texttemplate="%{text}",
            textfont=dict(size=8),
            hovertemplate="Country: %{y}<br>Year: %{x}<br>EV Sales Share: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="EV sales<br>share (%)", thickness=12, len=0.8),
        ))
        lay_hm = base_layout(max(420, len(hm_pivot) * 18))
        lay_hm.update({
            "xaxis": {**lay_hm.get("xaxis",{}), "side": "top"},
            "margin": dict(l=0, r=60, t=30, b=0),
        })
        fig_hm.update_layout(**lay_hm)
        st.plotly_chart(fig_hm, width="stretch", config=chart_config())
        st.markdown("---")

    # ── Regional breakdown + Top 10 ──────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Where is growth happening?")
        st.caption(f"EV sales by world region — {latest}")
        df_l2 = df_l.copy()
        df_l2["region"] = df_l2["country"].apply(_region)
        regional = df_l2.groupby("region")["ev_sales"].sum().sort_values(ascending=True)
        regional = regional[regional > 0]

        fig_reg = go.Figure(go.Bar(
            x=regional.values / 1e6, y=regional.index, orientation="h",
            marker_color=[NAVY if r != "Other" else SLATE for r in regional.index],
            text=(regional.values / 1e6).round(2).astype(str) + "M",
            textposition="outside", textfont=dict(size=9),
            hovertemplate="%{y}: %{x:.2f}M units<extra></extra>",
        ))
        lay_r = base_layout(300)
        lay_r.update({"xaxis": {**lay_r.get("xaxis",{}), "title": "Million units"},
                       "yaxis": {**lay_r.get("yaxis",{}), "title": ""},
                       "showlegend": False, "margin": {**lay_r.get("margin",{}), "r": 70}})
        fig_reg.update_layout(**lay_r)
        st.plotly_chart(fig_reg, width="stretch", config=chart_config())

    with col_r:
        st.markdown("### Top 10 fastest-growing markets")
        st.caption(f"YoY growth — {latest} — countries with >1,000 EV sales")
        top_gr = (
            df_l[df_l["ev_sales"] > 1000]
            .sort_values("growth_rate", ascending=False)
            .head(10)[["country","ev_sales","growth_rate"]]
            .sort_values("growth_rate")
        )
        fig_gr = go.Figure(go.Bar(
            x=top_gr["growth_rate"], y=top_gr["country"], orientation="h",
            marker_color=GREEN_MED,
            text=top_gr["growth_rate"].map(lambda v: f"{v:+.0f}%"),
            textposition="outside", textfont=dict(size=9),
            hovertemplate="%{y}: %{x:+.1f}% YoY<extra></extra>",
        ))
        lay_g = base_layout(300)
        lay_g.update({"xaxis": {**lay_g.get("xaxis",{}), "title": "YoY growth (%)"},
                       "yaxis": {**lay_g.get("yaxis",{}), "title": ""},
                       "showlegend": False, "margin": {**lay_g.get("margin",{}), "r": 70}})
        fig_gr.update_layout(**lay_g)
        st.plotly_chart(fig_gr, width="stretch", config=chart_config())

    st.markdown("---")

    # ── EV Sales Share leaders + vs world average ─────────────────────────────
    if "ev_sales_share" in df_l.columns and df_l["ev_sales_share"].notna().any():
        st.markdown("### EV sales share — what % of new cars are electric?")
        st.markdown(
            "EV *sales share* is the most actionable metric for investors and policymakers: "
            "of every 100 cars sold this year, how many are electric?"
        )
        world_avg_share = df_l["ev_sales_share"].mean()
        share_top = (
            df_l[df_l["ev_sales_share"].notna()]
            .groupby("country")["ev_sales_share"].mean()
            .nlargest(20).sort_values().reset_index()
        )
        norm = (share_top["ev_sales_share"] - share_top["ev_sales_share"].min()) / \
               (share_top["ev_sales_share"].max() - share_top["ev_sales_share"].min() + 1)
        fig_ss = go.Figure()
        fig_ss.add_trace(go.Bar(
            x=share_top["ev_sales_share"], y=share_top["country"], orientation="h",
            marker=dict(color=norm, colorscale=[[0,NAVY_PALE],[0.5,NAVY_LIGHT],[1,NAVY]],
                        showscale=False),
            text=share_top["ev_sales_share"].map(lambda v: f"{v:.1f}%"),
            textposition="outside", textfont=dict(size=9),
            hovertemplate="%{y}: %{x:.1f}% EV sales share<extra></extra>",
        ))
        fig_ss.add_vline(x=world_avg_share, line_dash="dot",
                          line_color=AMBER, line_width=1.5,
                          annotation_text=f"World avg {world_avg_share:.1f}%",
                          annotation_font_size=9)
        lay_ss = base_layout(420)
        lay_ss.update({"xaxis": {**lay_ss.get("xaxis",{}), "title": "EV sales share (%)"},
                        "yaxis": {**lay_ss.get("yaxis",{}), "title": ""},
                        "showlegend": False, "margin": {**lay_ss.get("margin",{}), "r": 60}})
        fig_ss.update_layout(**lay_ss)
        st.plotly_chart(fig_ss, width="stretch", config=chart_config())

    # ── EV Stock Share leaders ────────────────────────────────────────────────
    if "ev_stock_share" in df_l.columns and df_l["ev_stock_share"].notna().any():
        st.markdown("### EV stock share — how much of the fleet is already electric?")
        st.markdown(
            "Stock share is slower-moving than sales share — it reflects accumulated sales "
            "over years. Countries leading here have already turned over a significant part of their fleet."
        )
        world_avg_stock = df_l["ev_stock_share"].mean()
        share_top2 = (
            df_l[df_l["ev_stock_share"].notna()]
            .groupby("country")["ev_stock_share"].mean()
            .nlargest(15).sort_values().reset_index()
        )
        norm2 = (share_top2["ev_stock_share"] - share_top2["ev_stock_share"].min()) / \
                (share_top2["ev_stock_share"].max() - share_top2["ev_stock_share"].min() + 1)
        fig_sh = go.Figure(go.Bar(
            x=share_top2["ev_stock_share"], y=share_top2["country"], orientation="h",
            marker=dict(color=norm2, colorscale=[[0,NAVY_PALE],[1,NAVY]], showscale=False),
            text=share_top2["ev_stock_share"].map(lambda v: f"{v:.1f}%"),
            textposition="outside", textfont=dict(size=9),
            hovertemplate="%{y}: %{x:.1f}% EV stock share<extra></extra>",
        ))
        fig_sh.add_vline(x=world_avg_stock, line_dash="dot",
                          line_color=AMBER, line_width=1.5,
                          annotation_text=f"World avg {world_avg_stock:.1f}%",
                          annotation_font_size=9)
        lay_sh = base_layout(360)
        lay_sh.update({"xaxis": {**lay_sh.get("xaxis",{}), "title": "EV stock share (%)"},
                        "yaxis": {**lay_sh.get("yaxis",{}), "title": ""},
                        "showlegend": False, "margin": {**lay_sh.get("margin",{}), "r": 60}})
        fig_sh.update_layout(**lay_sh)
        st.plotly_chart(fig_sh, width="stretch", config=chart_config())

    st.markdown("---")

    # ── EVQTI preview ─────────────────────────────────────────────────────────
    st.markdown("### Which emerging markets have the most momentum?")
    st.markdown(
        "The **EVQTI** identifies countries with low current penetration but strong momentum — "
        "the markets most likely to accelerate next. Norway and Netherlands rank low despite "
        "being EV leaders, because their markets are maturing naturally."
    )
    try:
        rdf = GlobalRanking(df).compute_ranking()
        rdf["evqti_score"] = rdf["evqti_score"].round(3)
        rdf.columns = ["Country","Rank","EVQTI Score"]
        st.dataframe(rdf.head(10).reset_index(drop=True),
                     width="stretch", hide_index=True)
        st.caption("→ Go to **EVQTI Ranking** for the full ranking, IC validation and rolling backtest.")
    except Exception as e:
        st.warning(f"Ranking unavailable: {e}")
