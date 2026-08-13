# streamlit_pages/10_ETII.py
"""
ETII — EV Transition Intelligence Index

Combines two orthogonal signals into a single composite ranking:
  1. EVQTI  — momentum signal (early-stage markets grow fastest)
  2. World Bank Fundamentals Score — structural capacity to sustain the transition

These two signals are complementary, not redundant:
  - A country with high EVQTI but low fundamentals may be growing fast from a
    low base, but lacks the economic infrastructure to sustain the surge.
  - A country with high fundamentals but low EVQTI is wealthy and urbanised,
    but already penetrated — growth is slowing naturally.
  - The sweet spot (high both) = early-stage market with structural capacity
    to sustain the acceleration.

Design: equally-weighted composite (50/50) by default, adjustable via slider.
All component scores are min-max normalised to [0, 100] before combining.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models.scoring.evqti import EVQTI
from data.world_bank_client import get_country_fundamentals, get_fundamentals_score, _resolve_iso3
from visualization.theme import (
    base_layout, chart_config,
    NAVY, NAVY_LIGHT, NAVY_PALE, GREEN_MED, AMBER, GOLD, RED_DARK, SLATE,
    QUINTIL_COLORS,
)


@st.cache_data(ttl=3600, show_spinner="Computing ETII...")
def _compute_etii(df_hash: str, evqti_weight: float, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the ETII composite index.

    Resilient to World Bank network failures:
    - If all WB calls fail, ETII falls back to EVQTI-only ranking
    - Partial failures use cross-section median for missing scores
    - Each country is fetched independently; one failure never blocks others
    """
    # ── EVQTI ──
    model = EVQTI()
    model.fit(df)
    evqti_df = model.get_ranking()[["country", "evqti_score"]].copy()
    lo, hi = evqti_df["evqti_score"].min(), evqti_df["evqti_score"].max()
    evqti_df["evqti_norm"] = ((evqti_df["evqti_score"] - lo) / (hi - lo + 1e-9)) * 100

    # ── World Bank Fundamentals (one call per country, each isolated) ──
    countries  = evqti_df["country"].tolist()
    fund_rows  = []
    successful = 0

    for c in countries:
        iso3 = _resolve_iso3(c)
        if iso3 is None:
            fund_rows.append({"country": c, "fund_score": None})
            continue
        try:
            f     = get_country_fundamentals(c)
            score = get_fundamentals_score(f)
            fund_rows.append({"country": c, "fund_score": score})
            if score is not None:
                successful += 1
        except Exception:
            fund_rows.append({"country": c, "fund_score": None})

    fund_df = pd.DataFrame(fund_rows)

    # ── Combine — graceful degradation ──
    combined = evqti_df.merge(fund_df, on="country", how="left")

    valid_scores = combined["fund_score"].dropna()
    if len(valid_scores) == 0:
        # No WB data at all — use EVQTI only, fund_score = 50 (neutral)
        combined["fund_score"] = 50.0
        combined["wb_available"] = False
    else:
        combined["fund_score"]   = combined["fund_score"].fillna(valid_scores.median())
        combined["wb_available"] = True

    fund_wt = 1 - evqti_weight
    combined["etii_score"] = (
        combined["evqti_norm"] * evqti_weight +
        combined["fund_score"] * fund_wt
    ).round(2)

    combined["etii_rank"]  = combined["etii_score"].rank(ascending=False).astype(int)
    combined["evqti_rank"] = combined["evqti_norm"].rank(ascending=False).astype(int)
    combined["fund_rank"]  = combined["fund_score"].rank(ascending=False).astype(int)
    combined["wb_coverage_pct"] = successful / max(len(countries), 1) * 100

    return combined.sort_values("etii_score", ascending=False).reset_index(drop=True)


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "")
    st.markdown("## ETII — EV Transition Intelligence Index")
    st.markdown(
        "The EVQTI identifies *which* markets are early-stage and growing fastest. "
        "But early-stage growth in a country with weak economic infrastructure "
        "may not be sustainable. The **ETII** adds a second lens: "
        "*can this country actually sustain the transition?*"
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.info("**EVQTI** (50%)\nMomentum signal — early-stage markets with strong growth potential")
    col_b.info("**Fundamentals** (50%)\nWorld Bank — GDP, urbanization, economic capacity")
    col_c.success("**ETII** = Composite\nHighest-conviction early-stage opportunities")

    st.markdown("---")

    evqti_weight = st.slider(
        "EVQTI weight in composite (remainder goes to Fundamentals)",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="0.5 = equal weight. Higher = more momentum-driven. Lower = more fundamentals-driven."
    )
    fund_weight = round(1 - evqti_weight, 2)
    st.caption(f"EVQTI weight: {evqti_weight:.0%} · Fundamentals weight: {fund_weight:.0%}")

    # Use df identity as cache key (hash-free, just shape+columns)
    df_hash = f"{df.shape}_{evqti_weight}"
    with st.spinner("Fetching World Bank data and computing ETII..."):
        try:
            etii = _compute_etii(df_hash, evqti_weight, df)
        except Exception as e:
            st.error(f"ETII computation error: {e}")
            import traceback; st.code(traceback.format_exc())
            return

    if etii.empty:
        st.warning("Not enough data to compute ETII.")
        return

    wb_coverage = etii["wb_coverage_pct"].iloc[0] if "wb_coverage_pct" in etii.columns else 0
    wb_available = etii["wb_available"].iloc[0] if "wb_available" in etii.columns else False

    if not wb_available:
        st.warning(
            "⚠️ World Bank API unavailable — ETII computed with EVQTI only "
            "(Fundamentals Score set to neutral 50). "
            "The ranking still reflects momentum signal; fundamentals will load when the API is reachable."
        )
    else:
        st.caption(
            f"World Bank data fetched for {wb_coverage:.0f}% of countries. "
            "Countries without coverage use the cross-section median Fundamentals score."
        )
    st.markdown("---")

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    top3 = etii.head(3)
    st.markdown("### Top 3 highest-conviction opportunities")
    k1, k2, k3 = st.columns(3)
    for col, (_, row) in zip([k1, k2, k3], top3.iterrows()):
        col.metric(
            f"#{int(row['etii_rank'])} {row['country']}",
            f"ETII {row['etii_score']:.1f}/100",
            f"EVQTI #{int(row['evqti_rank'])} · Fundamentals #{int(row['fund_rank'])}",
            help="High ETII = early-stage momentum + strong economic capacity to sustain it."
        )

    st.markdown("---")

    # ── Main ranking chart ─────────────────────────────────────────────────────
    st.markdown("### Full ETII Ranking")
    n_show = st.slider("Countries to display", 10, len(etii), min(30, len(etii)))
    top_n  = etii.head(n_show).sort_values("etii_score")
    norm   = (top_n["etii_score"] - top_n["etii_score"].min()) / \
             (top_n["etii_score"].max() - top_n["etii_score"].min() + 1e-9)

    fig1 = go.Figure(go.Bar(
        x=top_n["etii_score"], y=top_n["country"],
        orientation="h",
        marker=dict(
            color=norm,
            colorscale=[[0, NAVY_PALE], [0.5, NAVY_LIGHT], [1.0, NAVY]],
            showscale=False,
        ),
        text=top_n["etii_score"].map(lambda v: f"{v:.1f}"),
        textposition="outside", textfont=dict(size=9),
        hovertemplate="%{y}<br>ETII: %{x:.1f}<extra></extra>",
    ))
    lay1 = base_layout(max(400, n_show * 22), f"ETII Score — Top {n_show} Countries")
    lay1.update({
        "xaxis":      {**lay1.get("xaxis",{}), "title": "ETII Score (0–100)"},
        "yaxis":      {**lay1.get("yaxis",{}), "title": ""},
        "margin":     {**lay1.get("margin",{}), "r": 70},
        "showlegend": False,
    })
    fig1.update_layout(**lay1)
    st.plotly_chart(fig1, use_container_width=True, config=chart_config())

    st.markdown("---")

    # ── EVQTI vs Fundamentals scatter ─────────────────────────────────────────
    st.markdown("### EVQTI vs Fundamentals — the opportunity matrix")
    st.markdown(
        "The most interesting countries are in the **top-right quadrant**: "
        "early-stage (high EVQTI) with strong economic capacity (high Fundamentals). "
        "Bottom-right = high-capacity but mature markets. "
        "Top-left = high momentum but structural fragility."
    )

    fig2 = go.Figure()

    # Quadrant shading
    mid_evqti = etii["evqti_norm"].median()
    mid_fund  = etii["fund_score"].median()

    # Sweet spot annotation
    fig2.add_shape(type="rect",
        x0=mid_evqti, x1=100, y0=mid_fund, y1=100,
        fillcolor="rgba(21,129,61,0.06)", line=dict(width=0),
    )

    fig2.add_trace(go.Scatter(
        x=etii["evqti_norm"], y=etii["fund_score"],
        mode="markers+text",
        text=etii["country"],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=etii["etii_score"] / 6 + 6,
            color=etii["etii_score"],
            colorscale=[[0, NAVY_PALE], [0.5, NAVY_LIGHT], [1.0, NAVY]],
            showscale=True,
            colorbar=dict(title="ETII", thickness=10, len=0.6),
            line=dict(width=0.5, color="white"),
            opacity=0.85,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "EVQTI norm: %{x:.1f}<br>"
            "Fundamentals: %{y:.1f}<br>"
            "<extra></extra>"
        ),
    ))

    fig2.add_vline(x=mid_evqti, line_dash="dot",
                   line_color="rgba(128,128,128,0.35)", line_width=1,
                   annotation_text="EVQTI median", annotation_font_size=9)
    fig2.add_hline(y=mid_fund, line_dash="dot",
                   line_color="rgba(128,128,128,0.35)", line_width=1,
                   annotation_text="Fundamentals median", annotation_font_size=9)

    # Quadrant labels
    for (x, y, txt) in [
        (92, 92, "★ Sweet spot"),
        (10, 92, "Mature markets"),
        (92, 10, "High momentum,\nfragile base"),
        (10, 10, "Limited potential"),
    ]:
        fig2.add_annotation(x=x, y=y, text=txt, showarrow=False,
                             font=dict(size=8, color="rgba(128,128,128,0.7)"))

    lay2 = base_layout(500)
    lay2.update({
        "xaxis":      {**lay2.get("xaxis",{}),
                        "title": "EVQTI Score (normalised 0–100) — higher = earlier stage",
                        "range": [-2, 102]},
        "yaxis":      {**lay2.get("yaxis",{}),
                        "title": "Fundamentals Score (World Bank) — higher = stronger capacity",
                        "range": [-2, 102]},
        "showlegend": False,
    })
    fig2.update_layout(**lay2)
    st.plotly_chart(fig2, use_container_width=True, config=chart_config())

    st.markdown("---")

    # ── Full table ─────────────────────────────────────────────────────────────
    st.markdown("### Full Table")
    disp = etii[["country","etii_rank","etii_score","evqti_norm","fund_score",
                  "evqti_rank","fund_rank"]].copy()
    disp.columns = ["Country","ETII Rank","ETII Score","EVQTI (norm)","Fundamentals",
                     "EVQTI Rank","Fund. Rank"]
    disp["ETII Score"]    = disp["ETII Score"].round(1)
    disp["EVQTI (norm)"]  = disp["EVQTI (norm)"].round(1)
    disp["Fundamentals"]  = disp["Fundamentals"].fillna("—").apply(
        lambda v: f"{v:.1f}" if isinstance(v, float) else v
    )
    st.dataframe(disp, use_container_width=True, hide_index=True)

    with st.expander("How is ETII calculated?"):
        st.markdown(f"""
**Component 1 — EVQTI (weight: {evqti_weight:.0%})**

The EV Quant Transition Index (v3) identifies early-stage markets using:
- EV stock share (t-1) — weight 48.5%
- EV sales share (t-1) — weight 51.5%

Validated IC = +0.237, IR = 0.99. Normalised to [0, 100] cross-section.

**Component 2 — World Bank Fundamentals (weight: {fund_weight:.0%})**

Equally-weighted composite of:
- GDP per capita (min-max normalised, $500–$80k range)
- Urbanisation rate (% of population)
- GDP growth (−5% to +10% range)
- CO2 per capita (partial weight, directionally ambiguous)

Countries without World Bank coverage use the cross-section median.

**Combination:**

ETII = EVQTI_norm × {evqti_weight:.0%} + Fundamentals × {fund_weight:.0%}

**Interpretation:**
- High ETII = early-stage momentum + structural capacity to sustain the transition
- Not a buy/sell signal — a research prioritisation tool
        """)
