# analytics/narrative_engine.py
"""
NarrativeEngine — Quantitative Research Intelligence Layer

Transforms model outputs into structured research insights in the style
of an institutional macro-quant desk: factual, evidence-based, forward-looking.

Consumes:
  - FactorEngine   (multi-factor scores + driver decomposition)
  - RegimeDetection (current and historical regimes)
  - RegimeAwareForecastEngine (conditional forecast)

Outputs per country:
  - headline      (one-line research note)
  - what_changed  (detected structural changes)
  - why           (quantitative drivers)
  - implication   (1-5 year forward view)
  - anomalies     (statistical outliers worth flagging)
  - bullets       (3-6 structured insight bullets)

Anti-pattern avoided: the engine does NOT generate generic LLM summaries.
Every insight is derived from a specific quantitative condition with
explicit thresholds — it is an analytical rule engine, not a text generator.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Thresholds for insight generation
_REGIME_CHANGE_LOOKBACK   = 3   # years to look back for regime change detection
_SCORE_SHIFT_THRESHOLD    = 10  # ETII/EVQTI rank shift considered significant
_INFRA_GAP_THRESHOLD      = 20  # EVs/charger above this = infrastructure concern
_INFRA_TARGET             = 10  # IEA reference target
_MOMENTUM_THRESHOLD       = 10  # pp momentum change considered significant
_GROWTH_SURGE_THRESHOLD   = 50  # % YoY growth above this = surge
_GROWTH_DECLINE_THRESHOLD =-20  # % YoY growth below this = decline signal


class NarrativeEngine:
    """
    Generates structured research narratives from quantitative model outputs.

    Usage
    -----
    engine = NarrativeEngine(factor_engine, regime_results, forecast_results, infra_df)
    country_narrative = engine.generate_country(country, df)
    global_narrative  = engine.generate_global(df)
    """

    def __init__(
        self,
        factor_engine=None,
        regime_results: Optional[dict] = None,
        forecast_results: Optional[dict] = None,
        infra_df: Optional[pd.DataFrame] = None,
    ):
        self.factor_engine    = factor_engine
        self.regime_results   = regime_results or {}
        self.forecast_results = forecast_results or {}
        # infra_df: real IEA ev_per_charger (country, year, ev_per_charger)
        self._infra = {}
        if infra_df is not None and not infra_df.empty:
            latest_infra = (
                infra_df.sort_values("year")
                .groupby("country")
                .last()
                .reset_index()
            )
            self._infra = dict(
                zip(latest_infra["country"], latest_infra["ev_per_charger"])
            )

    def generate_country(
        self, country: str, df: pd.DataFrame
    ) -> dict:
        """
        Generate full research narrative for a single country.

        Returns
        -------
        dict:
            headline, bullets, what_changed, why, implication, anomalies, score
        """
        country_df = df[df["country"] == country].sort_values("year")
        if country_df.empty:
            return {"headline": f"No data available for {country}", "bullets": []}

        bullets    = []
        anomalies  = []
        what_parts = []
        why_parts  = []
        impl_parts = []

        latest    = country_df.iloc[-1]
        prev      = country_df.iloc[-2] if len(country_df) > 1 else latest
        latest_yr = int(latest["year"])

        # ── 1. Growth signal ─────────────────────────────────────────────
        growth = latest.get("growth_rate", np.nan)
        if not np.isnan(growth):
            if growth >= _GROWTH_SURGE_THRESHOLD:
                bullets.append(
                    f"🚀 **EV sales surged {growth:+.0f}% YoY in {latest_yr}** — "
                    f"among the fastest growth rates in the dataset."
                )
                what_parts.append(f"sales surge (+{growth:.0f}% YoY)")
            elif growth <= _GROWTH_DECLINE_THRESHOLD:
                bullets.append(
                    f"⚠️ **Growth decelerated to {growth:+.0f}% in {latest_yr}**. "
                    f"May signal market saturation or policy withdrawal."
                )
                anomalies.append(f"Negative growth signal: {growth:.0f}% YoY")

        # ── 2. Momentum ──────────────────────────────────────────────────
        momentum = latest.get("momentum_3y", np.nan)
        if not np.isnan(momentum):
            if momentum >= _MOMENTUM_THRESHOLD:
                bullets.append(
                    f"📈 **Accelerating trend** — 3-year momentum at +{momentum:.1f}pp. "
                    f"Growth is compounding, not just a one-year event."
                )
                why_parts.append(f"compounding momentum (+{momentum:.1f}pp over 3 years)")
            elif momentum <= -_MOMENTUM_THRESHOLD:
                bullets.append(
                    f"📉 **Decelerating trend** — 3-year momentum at {momentum:.1f}pp. "
                    f"Growth rate is structurally slowing."
                )
                anomalies.append(f"Negative momentum: {momentum:.1f}pp")

        # ── 3. Penetration ───────────────────────────────────────────────
        stock_share = latest.get("ev_stock_share", np.nan)
        sales_share = latest.get("ev_sales_share", np.nan)
        if not np.isnan(stock_share):
            if stock_share < 2:
                bullets.append(
                    f"🌱 **Very early-stage market** — EV stock share at {stock_share:.1f}%. "
                    f"The growth ceiling is distant; Bass diffusion says the fastest growth lies ahead."
                )
                why_parts.append(f"low penetration ({stock_share:.1f}% stock share) = high remaining runway")
            elif stock_share > 25:
                bullets.append(
                    f"🏁 **Maturing market** — EV stock share at {stock_share:.1f}%. "
                    f"Early adopter phase complete; growth is naturally moderating."
                )
                why_parts.append(f"high penetration ({stock_share:.1f}% stock share) = natural slowdown")

        if not np.isnan(sales_share):
            bullets.append(
                f"🔋 **{sales_share:.1f}% of new cars sold in {latest_yr} were electric** "
                f"({'above' if sales_share > 10 else 'below'} the 10% mass-market threshold)."
            )

        # ── 4. Infrastructure gap ────────────────────────────────────────
        # Prefer real IEA data (self._infra); panel-derived value is unreliable
        ev_per_charger = self._infra.get(country, np.nan)
        if np.isnan(ev_per_charger):
            raw = latest.get("ev_per_charger", np.nan)
            # Sanity guard: derived column can be wildly wrong (ev_sales / 0 = inf)
            ev_per_charger = raw if (not np.isnan(raw) and 0 < raw < 10000) else np.nan
        if not np.isnan(ev_per_charger) and ev_per_charger > 0:
            if ev_per_charger > _INFRA_GAP_THRESHOLD:
                bullets.append(
                    f"⚡ **Infrastructure bottleneck** — {ev_per_charger:.0f} EVs per public charger "
                    f"(IEA target: {_INFRA_TARGET}). Charging network is not keeping pace with sales growth."
                )
                anomalies.append(f"Infrastructure gap: {ev_per_charger:.0f} EVs/charger (target: {_INFRA_TARGET})")
                why_parts.append(f"infrastructure constraint ({ev_per_charger:.0f} EVs/charger)")
            elif ev_per_charger <= _INFRA_TARGET:
                bullets.append(
                    f"✅ **Well-covered charging network** — {ev_per_charger:.0f} EVs per charger, "
                    f"at or below the IEA reference target of {_INFRA_TARGET}."
                )

        # ── 5. Regime ────────────────────────────────────────────────────
        regime = self.regime_results.get(country, {})
        if regime:
            current_regime = regime.get("current_regime", "")
            regime_change  = regime.get("regime_change", False)
            if regime_change:
                prev_regime = regime.get("previous_regime", "")
                bullets.append(
                    f"🔄 **Regime transition** — {country} moved from "
                    f"**{prev_regime}** to **{current_regime}** in {latest_yr}. "
                    f"This structural shift changes the forecast dynamics."
                )
                what_parts.append(f"regime transition ({prev_regime} → {current_regime})")
            else:
                bullets.append(
                    f"📊 **Adoption regime: {current_regime}** — "
                    f"{self._regime_description(current_regime)}"
                )

        # ── 6. Factor Engine drivers ─────────────────────────────────────
        if self.factor_engine is not None and self.factor_engine.fitted:
            try:
                drivers = self.factor_engine.get_drivers(country)
                if drivers.get("top_positive"):
                    top_drivers = [d["factor"].replace("_"," ").title()
                                   for d in drivers["top_positive"][:2]]
                    bullets.append(
                        f"🎯 **Key drivers**: {' and '.join(top_drivers)} are the main factors "
                        f"behind this ranking. Score: {drivers.get('score', 0):.1f}/100."
                    )
                    why_parts.extend([d["factor"].replace("_"," ") for d in drivers["top_positive"][:2]])
            except Exception:
                pass

        # ── 7. Forecast implication ──────────────────────────────────────
        forecast = self.forecast_results.get(country, {})
        if forecast and "baseline" in forecast:
            bl      = forecast["baseline"]
            narrative = forecast.get("narrative", {})
            if not bl.empty:
                horizon_end = int(bl["year"].iloc[-1])
                total_fcast = bl["median"].sum()
                impl_parts.append(
                    f"Median cumulative forecast: {total_fcast/1e6:.1f}M EV units over the forecast horizon "
                    f"(to {horizon_end}). {narrative.get('implication','')}"
                )
                bullets.append(f"📅 **{narrative.get('implication','No forecast available.')}**")

        # ── Assemble output ───────────────────────────────────────────────
        headline = self._build_headline(country, growth, stock_share, what_parts, latest_yr)

        return {
            "country":      country,
            "year":         latest_yr,
            "headline":     headline,
            "what_changed": "; ".join(what_parts) if what_parts else "No major structural change detected.",
            "why":          "; ".join(why_parts) if why_parts else "Driven by baseline penetration dynamics.",
            "implication":  " ".join(impl_parts) if impl_parts else "Forecast pending.",
            "anomalies":    anomalies,
            "bullets":      bullets[:6],  # max 6
            "n_bullets":    len(bullets),
        }

    def generate_global(self, df: pd.DataFrame) -> dict:
        """
        Generate global intelligence summary — top movers, anomalies, regional signals.
        """
        latest_yr = int(df["year"].max())
        df_l = df[df["year"] == latest_yr]

        bullets   = []
        anomalies = []

        # Top 3 surges
        surges = (
            df_l[df_l["growth_rate"] >= _GROWTH_SURGE_THRESHOLD]
            .sort_values("growth_rate", ascending=False)
            .head(3)
        )
        if not surges.empty:
            names = ", ".join(
                f"{r['country']} (+{r['growth_rate']:.0f}%)"
                for _, r in surges.iterrows()
            )
            bullets.append(f"🚀 **Fastest-growing markets in {latest_yr}**: {names}.")

        # Top 3 declines
        declines = df_l[df_l["growth_rate"] <= _GROWTH_DECLINE_THRESHOLD]
        if not declines.empty:
            names = ", ".join(declines.sort_values("growth_rate").head(3)["country"])
            bullets.append(
                f"⚠️ **Growth deceleration signals**: {names} showed significant YoY declines — "
                f"potential saturation or policy reversal."
            )
            anomalies.extend(declines["country"].tolist())

        # Infrastructure alarm countries
        if "ev_per_charger" in df_l.columns:
            infra_alarm = df_l[
                df_l["ev_per_charger"] > _INFRA_GAP_THRESHOLD * 2
            ].sort_values("ev_per_charger", ascending=False).head(3)
            if not infra_alarm.empty:
                names = ", ".join(
                    f"{r['country']} ({r['ev_per_charger']:.0f} EVs/charger)"
                    for _, r in infra_alarm.iterrows()
                )
                bullets.append(
                    f"⚡ **Infrastructure stress**: {names} are experiencing severe "
                    f"infrastructure gaps relative to their EV fleet."
                )

        # Global momentum
        global_growth = df_l["ev_sales"].sum()
        global_prev   = df[df["year"] == latest_yr - 1]["ev_sales"].sum()
        global_yoy    = (global_growth - global_prev) / global_prev * 100 if global_prev > 0 else 0
        bullets.append(
            f"🌍 **Global EV sales grew {global_yoy:+.1f}% in {latest_yr}** "
            f"({global_growth/1e6:.1f}M units total across {df_l['country'].nunique()} countries)."
        )

        # Median sales share
        if "ev_sales_share" in df_l.columns:
            median_ss = df_l["ev_sales_share"].median()
            bullets.append(
                f"🔋 **Global median EV sales share: {median_ss:.1f}%** — "
                f"meaning the median country sold {median_ss:.1f} EVs per 100 new cars in {latest_yr}."
            )

        return {
            "year":         latest_yr,
            "headline":     f"Global EV transition {latest_yr}: {global_yoy:+.1f}% growth, {global_growth/1e6:.1f}M units",
            "bullets":      bullets[:6],
            "anomalies":    anomalies,
            "n_countries":  df_l["country"].nunique(),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_headline(
        self,
        country: str,
        growth: float,
        stock_share: float,
        what_parts: list,
        year: int,
    ) -> str:
        if what_parts:
            return f"{country} — {what_parts[0].capitalize()} in {year}"
        if not np.isnan(growth):
            direction = "accelerating" if growth > 30 else "growing" if growth > 0 else "slowing"
            return f"{country} — EV market {direction} at {growth:+.1f}% YoY in {year}"
        return f"{country} — EV transition analysis {year}"

    def _regime_description(self, regime: str) -> str:
        descriptions = {
            "Emerging":     "early adopters dominating; broad market penetration has not yet begun.",
            "Expansion":    "adoption broadening beyond early adopters; infrastructure demand rising.",
            "Acceleration": "mainstream buyers entering; imitation dynamics dominating growth.",
            "Maturity":     "high penetration achieved; growth moderating as market approaches saturation.",
        }
        return descriptions.get(regime, "regime characteristics being established.")
