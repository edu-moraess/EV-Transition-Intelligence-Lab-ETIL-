# streamlit_pages/06_EVQTI_Ranking.py
"""
EVQTI Ranking — EV Quant Transition Index v3
Anti-leakage factor with IC, Rank IC, t-stat, IR and quintile analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from models.scoring.evqti import EVQTI
from visualization.theme import (
    base_layout, bar_color_signed,
    NAVY, NAVY_LIGHT, RED_DARK, GREEN_MED, AMBER, SLATE,
    QUINTIL_COLORS
)


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "")
    st.title("EVQTI Ranking")
    st.markdown(
        "**EV Quant Transition Index v3** — anti-leakage factor calibrated by empirical IC"
    )
    st.caption(
        f"Dataset: {dataset} · Signal at t-1 · Features: ev_stock_share · ev_sales_share "
        f"· Target: EV growth at t+1 · Weights calibrated by |IC|"
    )
    st.markdown("---")

    with st.spinner("Computing EVQTI v3..."):
        try:
            model = EVQTI()
            model.fit(df)
            rdf   = model.get_ranking()
            ic_df = model.get_ic_stats()
            dec   = model.get_decile_stats()
            s     = model.get_summary()
        except Exception as e:
            st.error(f"EVQTI error: {e}")
            import traceback; st.code(traceback.format_exc())
            return

    # ── Factor validation KPIs ────────────────────────────────────────────
    st.subheader("Factor Validation")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean Rank IC",       f"{s.get('ic_mean', 0):+.3f}",
              help="Spearman between EVQTI(t-1) and EV growth(t+1). |IC|>0.10 = robust signal.")
    c2.metric("IR (IC/σ)",          f"{s.get('ir', 0):.2f}",
              help="Information Ratio. IR>0.5 = consistent signal.")
    c3.metric("% years IC>0",       f"{s.get('pct_positive', 0):.0f}%",
              help="Proportion of years with positive IC.")
    c4.metric("% years significant", f"{s.get('pct_sig', 0):.0f}%",
              help="Years with p-value < 0.05.")
    c5.metric("Q5-Q1 spread",       f"{s.get('spread_q5_q1', 0):.1f}pp",
              help="Future growth difference between top and bottom quintile.")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # ── IC by year ────────────────────────────────────────────────────────
    with col_a:
        st.markdown("##### Rank IC by Year")
        if not ic_df.empty:
            fig_ic = go.Figure()
            fig_ic.add_trace(go.Bar(
                x=ic_df["year"],
                y=ic_df["rank_ic"],
                marker_color=bar_color_signed(ic_df["rank_ic"]),
                text=ic_df["rank_ic"].map(lambda v: f"{v:+.3f}"),
                textposition="outside",
                textfont=dict(size=9),
                name="Rank IC",
                hovertemplate="Year: %{x}<br>Rank IC: %{y:.3f}<br>t-stat: %{customdata:.2f}<extra></extra>",
                customdata=ic_df["t_stat"],
            ))
            fig_ic.add_hline(y=0.05,  line_dash="dot",
                             line_color="rgba(128,128,128,0.4)", line_width=1,
                             annotation_text="IC=0.05",
                             annotation_font_size=9)
            fig_ic.add_hline(y=-0.05, line_dash="dot",
                             line_color="rgba(128,128,128,0.4)", line_width=1)
            lay = base_layout(300)
            lay.update({"yaxis": {**lay.get("yaxis", {}),
                                   "title": "Rank IC (Spearman)",
                                   "range": [-0.45, 0.60]},
                         "showlegend": False})
            fig_ic.update_layout(**lay)
            st.plotly_chart(fig_ic, use_container_width=True)

            with st.expander("IC by year — detailed table"):
                show = ic_df[["year", "rank_ic", "t_stat", "p_value", "n", "significant"]].copy()
                show.columns = ["Year", "Rank IC", "t-stat", "p-value", "n", "Sig."]
                st.dataframe(show.sort_values("Year", ascending=False),
                             use_container_width=True, hide_index=True)

    # ── Quintile analysis ─────────────────────────────────────────────────
    with col_b:
        st.markdown("##### Quintile Analysis — Monotonic = Valid Factor")
        st.caption("Mean future YoY growth (t+1) by EVQTI quintile (t). Should increase left to right.")
        if not dec.empty:
            fig_dec = go.Figure()
            fig_dec.add_trace(go.Bar(
                x=dec["decile_label"],
                y=dec["avg_growth"],
                error_y=dict(type="data", array=dec["se"].values,
                             visible=True, thickness=1.2, width=4),
                marker_color=QUINTIL_COLORS[:len(dec)],
                text=dec["avg_growth"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate="%{x}<br>Avg growth: %{y:.1f}%<br>SE: ±%{error_y.array:.1f}pp<extra></extra>",
            ))
            lay2 = base_layout(300)
            lay2.update({"yaxis": {**lay2.get("yaxis", {}),
                                    "title": "Mean YoY growth (%) at t+1"},
                          "showlegend": False})
            fig_dec.update_layout(**lay2)
            st.plotly_chart(fig_dec, use_container_width=True)

    st.markdown("---")

    # ── Country ranking ───────────────────────────────────────────────────
    st.subheader("Country Ranking")
    n_show = st.slider("Countries to display", 10, len(rdf), min(30, len(rdf)))
    top_n  = rdf.head(n_show).sort_values("evqti_score")

    norm = (top_n["evqti_score"] - top_n["evqti_score"].min()) / \
           (top_n["evqti_score"].max() - top_n["evqti_score"].min() + 1e-9)

    fig_rank = go.Figure(go.Bar(
        x=top_n["evqti_score"],
        y=top_n["country"],
        orientation="h",
        marker=dict(
            color=norm,
            colorscale=[[0, "#c8d4e8"], [0.5, "#4d6ea8"], [1.0, NAVY]],
            showscale=False,
        ),
        text=top_n["evqti_score"].map(lambda v: f"{v:.3f}"),
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="%{y}<br>EVQTI: %{x:.4f}<extra></extra>",
    ))
    lay3 = base_layout(max(380, n_show * 22), f"EVQTI — Top {n_show}")
    lay3.update({"xaxis": {**lay3.get("xaxis", {}), "title": "EVQTI Score"},
                  "yaxis": {**lay3.get("yaxis", {}), "title": ""},
                  "margin": {**lay3.get("margin", {}), "r": 80},
                  "showlegend": False})
    fig_rank.update_layout(**lay3)
    st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("---")

    # ── EVQTI × CAGR × Market size ────────────────────────────────────────
    st.subheader("EVQTI × CAGR × Market Size")
    df_l    = df[df["year"] == df["year"].max()]
    df_meta = df_l.groupby("country").agg(
        cagr=("cagr", "mean"),
        ev_sales=("ev_sales", "sum"),
        growth_rate=("growth_rate", "mean")
    ).reset_index()
    sc_df = rdf.merge(df_meta, on="country", how="left")
    sc_df["ev_sales_log"] = np.log1p(sc_df["ev_sales"].fillna(0))
    sc_df["cagr"]         = sc_df["cagr"].fillna(sc_df["cagr"].median())

    fig_sc = go.Figure(go.Scatter(
        x=sc_df["evqti_score"], y=sc_df["cagr"],
        mode="markers",
        marker=dict(
            size=sc_df["ev_sales_log"] * 3,
            color=sc_df["evqti_score"],
            colorscale=[[0, "#c8d4e8"], [0.5, "#4d6ea8"], [1.0, NAVY]],
            showscale=True,
            colorbar=dict(title="EVQTI", thickness=10, len=0.6),
            line=dict(width=0.5, color="white"),
            opacity=0.85,
        ),
        text=sc_df["country"],
        hovertemplate="<b>%{text}</b><br>EVQTI: %{x:.3f}<br>CAGR: %{y:.1f}%<extra></extra>",
    ))
    lay4 = base_layout(420)
    lay4.update({"xaxis": {**lay4.get("xaxis", {}), "title": "EVQTI Score"},
                  "yaxis": {**lay4.get("yaxis", {}), "title": "CAGR (%)"},
                  "showlegend": False})
    fig_sc.update_layout(**lay4)
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # ── Full table ────────────────────────────────────────────────────────
    st.subheader("Full Table")
    full = rdf.merge(
        df_meta[["country", "cagr", "ev_sales", "growth_rate"]],
        on="country", how="left"
    )
    full["evqti_score"] = full["evqti_score"].round(4)
    full["cagr"]        = full["cagr"].fillna(0).round(1)
    full["ev_sales"]    = full["ev_sales"].fillna(0).round(0).astype("Int64")
    full["growth_rate"] = full["growth_rate"].fillna(0).round(1)
    full.columns        = ["Country", "Rank", "EVQTI Score", "CAGR (%)", "EV Sales", "YoY (%)"]
    st.dataframe(full, use_container_width=True, hide_index=True)

    # ── Methodology ───────────────────────────────────────────────────────
    with st.expander("Methodology — EVQTI v3"):
        st.markdown("""
**Economic rationale:** Countries in the early-stage of EV diffusion — low current penetration
with positive momentum — concentrate the highest future growth. This is Bass diffusion law
applied to the country cross-section: saturated markets grow slower; emerging markets grow faster.

**Factor construction (anti-leakage):**

| Feature (t-1) | Direction | Weight | Empirical IC |
|---|---|---|---|
| ev_stock_share | Negative (early-stage) | 48.5% | −0.259 (p<0.001) |
| ev_sales_share | Negative (early-stage) | 51.5% | −0.275 (p<0.001) |

*growth_rate_lag1 excluded: IC = −0.037, p = 0.36 — not statistically significant.*

**Normalisation pipeline:**
1. Winsorise features at 1%–99%
2. Create lags (t-1) on winsorised values
3. Re-winsorise lags at their own P1/P99
4. Cross-section z-score per year
5. Score = −(w₁·z₁ + w₂·z₂)
6. Winsorise target at 5%–95%

**IC interpretation:**

| Range | Meaning |
|---|---|
| IC < 0.05 | No useful signal |
| IC 0.05–0.10 | Weak signal |
| IC > 0.10 | Robust signal ✓ |
| IR > 0.5 | Consistent over time ✓ |

**Source:** IEA Global EV Outlook 2025/2026
        """)
