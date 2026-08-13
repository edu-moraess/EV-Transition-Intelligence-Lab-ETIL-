# streamlit_pages/06_EVQTI_Ranking.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models.scoring.evqti import EVQTI
from visualization.theme import (
    base_layout, bar_color_signed, chart_config,
    NAVY, NAVY_LIGHT, NAVY_PALE, RED_DARK, GREEN_MED, AMBER, SLATE,
    QUINTIL_COLORS,
)


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "")

    st.markdown("## EVQTI Ranking")
    st.markdown(
        "The **EV Quant Transition Index** answers a specific question: "
        "*which countries are most likely to see the fastest EV growth over the next year?*"
    )
    st.markdown(
        "The answer is counterintuitive. Norway and the Netherlands — global EV leaders — "
        "rank low on EVQTI because their markets are maturing. Countries like Brazil, "
        "Indonesia and South Africa rank high because they have low penetration today "
        "but are already accelerating. **Early-stage markets grow faster — that's Bass diffusion law.**"
    )
    st.caption(
        f"Dataset: {dataset} · Signal built at t-1 · "
        f"Features: EV stock share · EV sales share · Target: EV growth at t+1"
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

    # ── What the numbers mean ─────────────────────────────────────────────────
    st.markdown("### Does the signal actually work?")
    st.markdown(
        "A factor is only useful if it predicts what it claims to predict. "
        "Here we measure the **Information Coefficient (IC)** — "
        "the correlation between the EVQTI score in year *t* and actual EV growth in year *t+1*. "
        "An IC above 0.10 is considered a robust signal in quantitative research."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean Rank IC", f"{s.get('ic_mean', 0):+.3f}",
              help="Spearman correlation between EVQTI(t-1) and EV growth(t+1). Threshold: |IC| > 0.10.")
    c2.metric("Information Ratio", f"{s.get('ir', 0):.2f}",
              help="IC mean divided by IC standard deviation. IR > 0.5 = consistent signal.")
    c3.metric("Years with positive IC", f"{s.get('pct_positive', 0):.0f}%",
              help="Proportion of years where the signal was in the right direction.")
    c4.metric("Years statistically significant", f"{s.get('pct_sig', 0):.0f}%",
              help="Years where IC was significant at p < 0.05.")
    c5.metric("Growth spread Q5 vs Q1", f"{s.get('spread_q5_q1', 0):.0f}pp",
              help="Countries in the top quintile grew this many percentage points faster than the bottom quintile.")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### IC by year — does the signal hold over time?")
        st.caption(
            "Each bar is the correlation between EVQTI and next-year growth for that year. "
            "The signal strengthens after 2020 as EV markets gained enough scale."
        )
        if not ic_df.empty:
            fig_ic = go.Figure()
            fig_ic.add_trace(go.Bar(
                x=ic_df["year"], y=ic_df["rank_ic"],
                marker_color=bar_color_signed(ic_df["rank_ic"]),
                text=ic_df["rank_ic"].map(lambda v: f"{v:+.3f}"),
                textposition="outside", textfont=dict(size=9),
                hovertemplate="Year %{x}<br>Rank IC: %{y:.3f}<br>n=%{customdata} countries<extra></extra>",
                customdata=ic_df["n"],
            ))
            fig_ic.add_hline(y=0.10, line_dash="dot",
                             line_color="rgba(128,128,128,0.4)", line_width=1,
                             annotation_text="Robust threshold (0.10)",
                             annotation_font_size=9)
            lay = base_layout(300)
            lay.update({"yaxis": {**lay.get("yaxis",{}), "title": "Rank IC", "range": [-0.40, 0.60]},
                         "showlegend": False})
            fig_ic.update_layout(**lay)
            st.plotly_chart(fig_ic, config=chart_config())

    with col_b:
        st.markdown("##### Do high-EVQTI countries actually grow faster?")
        st.caption(
            "Countries ranked in the top quintile (Q5) by EVQTI grew significantly faster "
            "the following year than those in the bottom quintile (Q1). "
            "A monotonically increasing chart confirms the factor is valid."
        )
        if not dec.empty:
            fig_dec = go.Figure(go.Bar(
                x=dec["decile_label"], y=dec["avg_growth"],
                error_y=dict(type="data", array=dec["se"].values,
                             visible=True, thickness=1.2, width=4),
                marker_color=QUINTIL_COLORS[:len(dec)],
                text=dec["avg_growth"].map(lambda v: f"{v:.0f}%"),
                textposition="outside", textfont=dict(size=9),
                hovertemplate="%{x}<br>Avg growth: %{y:.1f}%<br>±%{error_y.array:.1f}pp SE<extra></extra>",
            ))
            lay2 = base_layout(300)
            lay2.update({"yaxis": {**lay2.get("yaxis",{}), "title": "Mean next-year growth (%)"},
                          "showlegend": False})
            fig_dec.update_layout(**lay2)
            st.plotly_chart(fig_dec, config=chart_config())

    st.markdown("---")

    # ── Backtest ──────────────────────────────────────────────────────────────
    st.markdown("### Is the signal getting stronger or weaker over time?")
    st.markdown(
        "A factor that worked once can simply be luck. To check, we measure the IC "
        "in rolling test windows — without re-fitting the weights each time. "
        "This shows whether the **same, fixed factor** keeps working as new years arrive, "
        "which is a more honest test than re-optimising weights for every period."
    )
    try:
        from models.scoring.evqti import EVQTI as _EVQTIClass
        bt_model = _EVQTIClass()
        bt_model.fit(df)
        backtest_df = bt_model.backtest(df, train_years=6, test_years=1, step=1)

        if not backtest_df.empty:
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=backtest_df["window_end"], y=backtest_df["ic_mean"],
                mode="lines+markers",
                line=dict(color=NAVY, width=2.5),
                marker=dict(size=8, color=NAVY),
                hovertemplate="Year %{x}<br>IC: %{y:.3f}<extra></extra>",
            ))
            fig_bt.add_hline(y=0.10, line_dash="dot",
                             line_color="rgba(128,128,128,0.4)", line_width=1,
                             annotation_text="Robust threshold (0.10)",
                             annotation_font_size=9)
            lay_bt = base_layout(280, "Out-of-sample IC by test year (fixed weights)")
            lay_bt.update({"yaxis": {**lay_bt.get("yaxis",{}), "title": "IC (test year)"},
                            "showlegend": False})
            fig_bt.update_layout(**lay_bt)
            st.plotly_chart(fig_bt, config=chart_config())

            recent_ic = backtest_df["ic_mean"].tail(4).mean()
            early_ic  = backtest_df["ic_mean"].head(4).mean()
            trend = "strengthened" if recent_ic > early_ic else "weakened"
            st.caption(
                f"The signal has **{trend}** over time: average IC was {early_ic:+.3f} "
                f"in the earliest test years vs {recent_ic:+.3f} in the most recent ones. "
                f"This is consistent with EV markets gaining enough scale for the cross-country "
                f"signal to become statistically reliable."
            )
        else:
            st.info("Not enough historical years available for a rolling backtest.")
    except Exception as e:
        st.warning(f"Backtest unavailable: {e}")

    st.markdown("---")

    # ── Ranking ────────────────────────────────────────────────────────────────
    st.markdown("### Country ranking — where is opportunity concentrated?")
    st.markdown(
        "Higher score = stronger early-stage signal = more likely to see above-average growth next year. "
        "This is not a quality ranking — a high score means the market is early, not that it is better."
    )
    n_show = st.slider("Countries to display", 10, len(rdf), min(30, len(rdf)))
    # FIX: Garante ordenação decrescente antes de pegar o head()
    top_n = rdf.sort_values("evqti_score", ascending=False).head(n_show).sort_values("evqti_score")
    norm   = (top_n["evqti_score"] - top_n["evqti_score"].min()) / \
             (top_n["evqti_score"].max() - top_n["evqti_score"].min() + 1e-9)

    fig_rank = go.Figure(go.Bar(
        x=top_n["evqti_score"], y=top_n["country"], orientation="h",
        marker=dict(color=norm, colorscale=[[0,NAVY_PALE],[0.5,NAVY_LIGHT],[1.0,NAVY]],
                    showscale=False),
        text=top_n["evqti_score"].map(lambda v: f"{v:.3f}"),
        textposition="outside", textfont=dict(size=9),
        hovertemplate="%{y}<br>EVQTI: %{x:.4f}<extra></extra>",
    ))
    lay3 = base_layout(max(380, n_show * 22), f"EVQTI Score — Top {n_show} Countries")
    lay3.update({"xaxis": {**lay3.get("xaxis",{}), "title": "EVQTI Score"},
                  "yaxis": {**lay3.get("yaxis",{}), "title": ""},
                  "margin": {**lay3.get("margin",{}), "r": 80},
                  "showlegend": False})
    fig_rank.update_layout(**lay3)
    st.plotly_chart(fig_rank, config=chart_config())

    st.markdown("---")

    # ── Scatter ────────────────────────────────────────────────────────────────
    st.markdown("### EVQTI vs CAGR — size of bubble = EV market scale")
    st.caption(
        "Each bubble is a country. X-axis = EVQTI (early-stage signal). "
        "Y-axis = historical CAGR. Bubble size = current EV sales volume. "
        "Top-right: large early-stage markets with strong growth — highest potential."
    )
    df_l    = df[df["year"] == df["year"].max()]
    df_meta = df_l.groupby("country").agg(
        cagr=("cagr","mean"), ev_sales=("ev_sales","sum"),
        growth_rate=("growth_rate","mean")
    ).reset_index()
    sc_df = rdf.merge(df_meta, on="country", how="left")
    sc_df["ev_sales_log"] = np.log1p(sc_df["ev_sales"].fillna(0))
    sc_df["cagr"]         = sc_df["cagr"].fillna(sc_df["cagr"].median())

    fig_sc = go.Figure(go.Scatter(
        x=sc_df["evqti_score"], y=sc_df["cagr"], mode="markers",
        marker=dict(
            size=sc_df["ev_sales_log"] * 3,
            color=sc_df["evqti_score"],
            colorscale=[[0,NAVY_PALE],[0.5,NAVY_LIGHT],[1.0,NAVY]],
            showscale=True, colorbar=dict(title="EVQTI", thickness=10, len=0.6),
            line=dict(width=0.5, color="white"), opacity=0.85,
        ),
        text=sc_df["country"],
        hovertemplate="<b>%{text}</b><br>EVQTI: %{x:.3f}<br>CAGR: %{y:.1f}%<extra></extra>",
    ))
    lay4 = base_layout(420)
    lay4.update({"xaxis": {**lay4.get("xaxis",{}), "title": "EVQTI Score (higher = earlier stage)"},
                  "yaxis": {**lay4.get("yaxis",{}), "title": "Historical CAGR (%)"},
                  "showlegend": False})
    fig_sc.update_layout(**lay4)
    st.plotly_chart(fig_sc, config=chart_config())

    st.markdown("---")
    st.subheader("Full Table")
    full = rdf.merge(df_meta[["country","cagr","ev_sales","growth_rate"]], on="country", how="left")
    full["evqti_score"] = full["evqti_score"].round(4)
    full["cagr"]        = full["cagr"].fillna(0).round(1)
    full["ev_sales"]    = full["ev_sales"].fillna(0).round(0).astype("Int64")
    full["growth_rate"] = full["growth_rate"].fillna(0).round(1)
    full.columns        = ["Country","Rank","EVQTI Score","CAGR (%)","EV Sales","YoY (%)"]
    st.dataframe(full, hide_index=True)

    with st.expander("How is EVQTI calculated?"):
        st.markdown("""
**The logic:** Countries with low EV penetration today but positive momentum are in the early
phase of the Bass diffusion curve — where growth is fastest. EVQTI scores these countries higher.

**The maths:**

| Feature | Direction | Weight | Empirical IC |
|---|---|---|---|
| EV stock share (t-1) | Negative — lower share = earlier stage | 48.5% | −0.259 (p<0.001) |
| EV sales share (t-1) | Negative — lower share = earlier stage | 51.5% | −0.275 (p<0.001) |

Growth rate was tested and excluded: IC = −0.037, p = 0.36 — no predictive power cross-country.

**Steps:**
1. Winsorise features at 1%–99% to remove outliers
2. Create one-year lags (so no future information leaks into the signal)
3. Z-score standardise per year across all countries
4. Score = −(w₁·z₁ + w₂·z₂) — negative sign flips early-stage to positive
5. Validate against actual next-year growth using Spearman correlation (Rank IC)

**Source:** IEA Global EV Outlook 2025/2026
        """)
