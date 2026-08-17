# streamlit_pages/08_Infrastructure.py
"""
Infrastructure Intelligence — EV charging network analysis.

The EV transition isn't just about vehicles — it depends critically on
the charging network. A country with fast-growing EV sales but stagnant
infrastructure will eventually hit a ceiling. This page tracks the build-out.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ingestion.loader import IEALoader
from visualization.theme import (
    base_layout, bar_color_signed, chart_config,
    NAVY, NAVY_LIGHT, NAVY_PALE, RED_DARK, GREEN_MED, AMBER, GOLD, SLATE,
)

_CHARGER_COLORS = {
    "Publicly available fast":       AMBER,
    "Publicly available slow":       NAVY,
    "Publicly available ultra-fast": GOLD,
}


@st.cache_data(ttl=600, show_spinner=False)
def _load_infra(dataset_year: int) -> dict:
    loader = IEALoader()
    agg    = loader.get_aggregator(dataset_year, countries_only=True)
    return {
        "cp":  agg.charging_points(category="Historical"),
        "ct":  agg.charging_by_type(category="Historical"),
        "epc": agg.ev_per_charger(category="Historical"),
    }


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "2026")
    try:
        year_int = int(dataset) if dataset in ("2025", "2026") else 2026
    except Exception:
        year_int = 2026

    st.markdown("## Infrastructure Intelligence")
    st.markdown(
        "A country can have strong EV policy, but limited charging access can slow "
        "adoption. This page tracks the infrastructure build-out: how many public "
        "charging points exist, how quickly the network is growing, and how the "
        "network compares with the size of the EV fleet."
    )
    st.caption(f"Source: IEA Global EV Outlook {year_int} · Public charging points; EVs per public charging point use the study’s defined EV fleet.")
    st.markdown("---")

    with st.spinner("Loading infrastructure data..."):
        try:
            data = _load_infra(year_int)
        except Exception as e:
            st.error(f"Infrastructure data unavailable: {e}")
            return

    cp  = data["cp"]
    ct  = data["ct"]
    epc = data["epc"]

    if cp.empty:
        st.warning("No charging point data available for this dataset.")
        return

    latest_year = int(cp["year"].max())
    prev_year   = latest_year - 1
    cp_l  = cp[cp["year"] == latest_year]
    cp_p  = cp[cp["year"] == prev_year]
    epc_l = epc[epc["year"] == latest_year]

    # ── Global KPIs ──────────────────────────────────────────────────────────
    total_cp  = cp_l["charging_points"].sum()
    total_cp_prev = cp_p["charging_points"].sum() if not cp_p.empty else 0
    yoy_cp    = (total_cp - total_cp_prev) / total_cp_prev * 100 if total_cp_prev > 0 else 0
    n_ctries  = cp_l["country"].nunique()
    median_epc = epc_l["ev_per_charger"].median() if not epc_l.empty else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Public charging points — {latest_year}",
              f"{total_cp/1e6:.2f}M",
              f"{yoy_cp:+.1f}% vs {prev_year}",
              help="Total publicly available charging points across all tracked countries.")
    c2.metric("Countries tracked", str(n_ctries),
              help="Countries with IEA charging point data.")
    c3.metric("Median EVs per public charging point",
              f"{median_epc:.0f}" if median_epc else "N/A",
              help="Lower = greater coverage relative to fleet size. This study uses a reference level of ~10 EVs per public charging point.")
    # Fastest growing infra country
    merged = cp_l.merge(cp_p[["country","charging_points"]], on="country", how="inner",
                        suffixes=("","_prev"))
    if not merged.empty:
        merged["cp_yoy"] = (merged["charging_points"] - merged["charging_points_prev"]) / \
                            merged["charging_points_prev"] * 100
        top_infra = merged.nlargest(1, "cp_yoy").iloc[0]
        c4.metric("Fastest-growing public charging network",
                  top_infra["country"],
                  f"+{top_infra['cp_yoy']:.0f}% public charging points YoY")

    st.markdown("---")

    # ── Top 15 countries by total chargers ────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Who has the most public charging points?")
        st.caption(f"Total public charging points — {latest_year}")
        top15 = cp_l.nlargest(15, "charging_points").sort_values("charging_points")
        norm  = (top15["charging_points"] - top15["charging_points"].min()) / \
                (top15["charging_points"].max() - top15["charging_points"].min() + 1)
        fig1 = go.Figure(go.Bar(
            x=top15["charging_points"] / 1e3, y=top15["country"],
            orientation="h",
            marker=dict(color=norm, colorscale=[[0, NAVY_PALE],[0.5, NAVY_LIGHT],[1.0, NAVY]],
                        showscale=False),
            text=top15["charging_points"].map(lambda v: f"{v/1e3:.0f}k"),
            textposition="outside", textfont=dict(size=9),
            hovertemplate="%{y}: %{x:.1f}k public charging points<extra></extra>",
        ))
        lay1 = base_layout(380)
        lay1.update({"xaxis": {**lay1.get("xaxis",{}), "title": "Thousand public charging points"},
                      "yaxis": {**lay1.get("yaxis",{}), "title": ""},
                      "showlegend": False, "margin": {**lay1.get("margin",{}), "r": 60}})
        fig1.update_layout(**lay1)
        st.plotly_chart(fig1, config=chart_config())

    with col_r:
        st.markdown("### Who has the fewest EVs per public charging point?")
        st.caption(
            f"EVs per public charging point — {latest_year}  \n"
            "Lower = fewer EVs per public charging point; this indicates greater coverage relative to fleet size"
        )
        epc_top = epc_l.nsmallest(15, "ev_per_charger").sort_values("ev_per_charger", ascending=False)
        fig2 = go.Figure(go.Bar(
            x=epc_top["ev_per_charger"], y=epc_top["country"],
            orientation="h",
            marker_color=[GREEN_MED if v <= 15 else AMBER if v <= 30 else RED_DARK
                          for v in epc_top["ev_per_charger"]],
            text=epc_top["ev_per_charger"].map(lambda v: f"{v:.0f}"),
            textposition="outside", textfont=dict(size=9),
            hovertemplate="%{y}: %{x:.0f} EVs per public charging point<extra></extra>",
        ))
        fig2.add_vline(x=10, line_dash="dot", line_color=GREEN_MED, line_width=1.5,
                       annotation_text="Reference level (10)", annotation_font_size=9)
        lay2 = base_layout(380)
        lay2.update({"xaxis": {**lay2.get("xaxis",{}), "title": "EVs per public charging point (lower = more coverage)"},
                      "yaxis": {**lay2.get("yaxis",{}), "title": ""},
                      "showlegend": False, "margin": {**lay2.get("margin",{}), "r": 60}})
        fig2.update_layout(**lay2)
        st.plotly_chart(fig2, config=chart_config())

    st.markdown("---")

    # ── Charger growth trend — global ─────────────────────────────────────────
    st.markdown("### How quickly is the global public charging network expanding?")
    st.markdown(
        "Public charging capacity and accessibility need to keep pace with EV demand "
        "to avoid infrastructure bottlenecks. The chart below shows annual growth "
        "in the total number of public charging points across all tracked countries."
    )
    cp_global = cp.groupby("year")["charging_points"].sum().reset_index()
    cp_global["yoy"] = cp_global["charging_points"].pct_change() * 100

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=cp_global["year"], y=cp_global["charging_points"] / 1e6,
        name="Total public charging points (M)", marker_color=NAVY, opacity=0.85,
        hovertemplate="Year %{x}: %{y:.2f}M public charging points<extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=cp_global["year"], y=cp_global["yoy"],
        name="YoY growth (%)", mode="lines+markers",
        line=dict(color=GOLD, width=2.5),
        marker=dict(size=7, color=GOLD),
        yaxis="y2",
        hovertemplate="Year %{x}: %{y:.1f}% YoY<extra></extra>",
    ))
    lay3 = base_layout(340)
    lay3.update({
        "yaxis":  dict(title="Million public charging points", gridcolor="rgba(128,128,128,0.12)"),
        "yaxis2": dict(title="YoY growth (%)", overlaying="y", side="right", showgrid=False),
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                       bgcolor="rgba(0,0,0,0)"),
        "barmode": "overlay",
    })
    fig3.update_layout(**lay3)
    st.plotly_chart(fig3, config=chart_config())

    st.markdown("---")

    # ── Country deep-dive ─────────────────────────────────────────────────────
    st.markdown("### Country deep-dive")
    countries_with_data = sorted(cp["country"].unique())
    selected = st.selectbox("Select a country", countries_with_data,
                             index=countries_with_data.index("Germany")
                             if "Germany" in countries_with_data else 0)

    cp_c   = cp[cp["country"] == selected].sort_values("year")
    ct_c   = ct[ct["country"] == selected].sort_values("year")
    epc_c  = epc[epc["country"] == selected].sort_values("year")

    if not cp_c.empty:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"##### Total public charging points — {selected}")
            fig4 = go.Figure(go.Bar(
                x=cp_c["year"], y=cp_c["charging_points"] / 1e3,
                marker_color=NAVY, opacity=0.85,
                hovertemplate="%{x}: %{y:.1f}k public charging points<extra></extra>",
            ))
            lay4 = base_layout(260)
            lay4.update({"yaxis": {**lay4.get("yaxis",{}), "title": "Thousand public charging points"},
                          "showlegend": False})
            fig4.update_layout(**lay4)
            st.plotly_chart(fig4, config=chart_config())

        with col_b:
            st.markdown(f"##### EVs per public charging point — {selected}")
            st.caption("Lower means fewer EVs per public charging point and greater coverage relative to fleet size.")
            if not epc_c.empty:
                fig5 = go.Figure(go.Scatter(
                    x=epc_c["year"], y=epc_c["ev_per_charger"],
                    mode="lines+markers",
                    line=dict(color=AMBER, width=2.5),
                    marker=dict(size=7, color=AMBER),
                    fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                    hovertemplate="%{x}: %{y:.0f} EVs per public charging point<extra></extra>",
                ))
                fig5.add_hline(y=10, line_dash="dot", line_color=GREEN_MED, line_width=1.5,
                               annotation_text="Reference level", annotation_font_size=9)
                lay5 = base_layout(260)
                lay5.update({"yaxis": {**lay5.get("yaxis",{}), "title": "EVs per public charging point"},
                              "showlegend": False})
                fig5.update_layout(**lay5)
                st.plotly_chart(fig5, config=chart_config())

        if not ct_c.empty:
            st.markdown(f"##### Public charging-point mix by type — {selected}")
            st.caption("Fast and ultra-fast chargers support longer-distance travel and reduce charging time.")
            types = ct_c["charger_type"].unique()
            fig6 = go.Figure()
            for t in types:
                sub = ct_c[ct_c["charger_type"] == t]
                short_label = t.replace("Publicly available ", "").title()
                fig6.add_trace(go.Bar(
                    x=sub["year"], y=sub["charging_points"] / 1e3,
                    name=short_label,
                    marker_color=_CHARGER_COLORS.get(t, SLATE),
                    opacity=0.88,
                    hovertemplate=f"{short_label} · %{{x}}: %{{y:.1f}}k<extra></extra>",
                ))
            lay6 = base_layout(280)
            lay6.update({"yaxis": {**lay6.get("yaxis",{}), "title": "Thousand chargers"},
                          "barmode": "stack"})
            fig6.update_layout(**lay6)
            st.plotly_chart(fig6, config=chart_config())

        # Infrastructure vs EV sales correlation
        # NOTE: this page's `df` is not guaranteed to carry the EV sales dataset
        # (see sidebar notice "This page does not require the EV dataset."),
        # so guard against a missing/differently-scoped df instead of assuming
        # "country"/"ev_sales" are always present. Fixes KeyError: 'country'.
        if "country" in df.columns and "ev_sales" in df.columns:
            ev_c = df[df["country"] == selected][["year", "ev_sales"]].copy()
            merged_c = cp_c.merge(ev_c, on="year", how="inner")
        else:
            merged_c = pd.DataFrame()

        if len(merged_c) >= 4:
            st.markdown(f"##### Infrastructure vs EV sales — {selected}")
            st.caption(
                "If public charging points grow slower than EV demand, relative coverage may deteriorate. "
                "If the network expands faster, coverage may improve."
            )
            merged_c["cp_norm"] = merged_c["charging_points"] / merged_c["charging_points"].iloc[0]
            merged_c["ev_norm"] = merged_c["ev_sales"] / merged_c["ev_sales"].iloc[0]
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(
                x=merged_c["year"], y=merged_c["cp_norm"],
                name="Public charging-point growth (indexed)", mode="lines+markers",
                line=dict(color=NAVY, width=2.5), marker=dict(size=6),
                hovertemplate="%{x}: %{y:.1f}x baseline<extra></extra>",
            ))
            fig7.add_trace(go.Scatter(
                x=merged_c["year"], y=merged_c["ev_norm"],
                name="EV sales growth (indexed)", mode="lines+markers",
                line=dict(color=GOLD, width=2.5), marker=dict(size=6),
                hovertemplate="%{x}: %{y:.1f}x baseline<extra></extra>",
            ))
            fig7.add_hline(y=1, line_dash="dot",
                           line_color="rgba(128,128,128,0.4)", line_width=1)
            lay7 = base_layout(280, "Indexed growth (first year = 1.0x)")
            lay7.update({"yaxis": {**lay7.get("yaxis",{}), "title": "Growth vs baseline"},
                          "xaxis": {**lay7.get("xaxis",{}), "title": "Year"}})
            fig7.update_layout(**lay7)
            st.plotly_chart(fig7, config=chart_config())

        # ── Infrastructure projection ──────────────────────────────────────────
        st.markdown(f"##### Infrastructure forecast — when does {selected} reach the 10-EV reference level?")
        st.caption(
            "Linear extrapolation of the recent EVs-per-public-charging-point trend. "
            "Shows when the country is projected to reach the study’s reference level of 10 EVs per public charging point. "
            "⚠️ Linear extrapolation only — policy changes, charging-point deployment or shifts in EV demand can change this substantially."
        )
        if len(epc_c) >= 3 and not epc_c.empty:
            from scipy.stats import linregress
            recent_epc = epc_c.tail(4)
            slope, intercept, r2, _, _ = linregress(
                recent_epc["year"].values,
                recent_epc["ev_per_charger"].values
            )
            r2 = r2 ** 2  # linregress returns r, not r²

            # Project 10 years forward
            last_yr = int(epc_c["year"].max())
            future_years = list(range(last_yr + 1, last_yr + 11))
            proj_vals = [max(slope * y + intercept, 0) for y in future_years]

            # Find when the study reference level of 10 is reached
            target = 10
            target_year = None
            for y, v in zip(future_years, proj_vals):
                if v <= target:
                    target_year = y
                    break

            fig8 = go.Figure()
            fig8.add_trace(go.Scatter(
                x=epc_c["year"], y=epc_c["ev_per_charger"],
                mode="lines+markers", name="Historical",
                line=dict(color=AMBER, width=2.5), marker=dict(size=7, color=AMBER),
                hovertemplate="%{x}: %{y:.0f} EVs per public charging point<extra></extra>",
            ))
            fig8.add_trace(go.Scatter(
                x=future_years, y=proj_vals,
                mode="lines+markers", name="Projected (linear)",
                line=dict(color=AMBER, width=2, dash="dash"),
                marker=dict(size=6, color=AMBER, symbol="diamond"),
                hovertemplate="%{x}: %{y:.0f} EVs per public charging point (projected)<extra></extra>",
            ))
            fig8.add_hline(y=target, line_dash="dot", line_color=GREEN_MED, line_width=1.5,
                           annotation_text="Reference level (10)",
                           annotation_font_size=9)
            if target_year:
                fig8.add_vline(x=target_year, line_dash="dot",
                               line_color=GREEN_MED, line_width=1,
                               annotation_text=f"Reaches reference level ~{target_year}",
                               annotation_font_size=9)

            lay8 = base_layout(280)
            lay8.update({"yaxis": {**lay8.get("yaxis",{}), "title": "EVs per public charging point (lower = more coverage)"},
                          "xaxis": {**lay8.get("xaxis",{}), "title": "Year"}})
            fig8.update_layout(**lay8)
            st.plotly_chart(fig8, config=chart_config())

            if target_year:
                current_val = float(epc_c["ev_per_charger"].iloc[-1])
                st.caption(
                    f"{selected} currently has **{current_val:.0f} EVs per public charging point**. "
                    f"At the current trend (slope {slope:+.1f} EVs per public charging point per year), it is projected to reach "
                    f"the study’s reference level of 10 EVs per public charging point around **{target_year}**."
                )
            else:
                st.caption(
                    f"At the current trend, {selected} does not reach the study’s reference level of 10 EVs per public charging point "
                    f"within the 10-year projection window. Infrastructure build-out needs to accelerate."
                )
    else:
        st.info(f"No charging infrastructure data available for {selected}.")
