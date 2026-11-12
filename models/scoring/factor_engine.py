# models/scoring/factor_engine.py
"""
FactorEngine — Multi-factor EV Transition Model

Expands EVQTI (2-factor) into a full multi-dimensional factor model
using all available lagged features. Maintains backward compatibility
with EVQTI v3 scores while adding explanatory layers.

Factor structure (all features at t-1, cross-section z-scored per year):

  1. Penetration signal   (EVQTI core)      — ev_stock_share_lag1, ev_sales_share_lag1
  2. Infrastructure gap   (NEW)             — ev_per_charger_lag1 (higher = worse infra)
  3. Energy dependency    (NEW)             — gasoline_price_lag1 (higher = more EV incentive)
  4. Macro capacity       (NEW)             — gdp_lag1, population_lag1 (affluence + scale)
  5. Momentum             (NEW)             — momentum_3y_lag1 (acceleration of growth)

Anti-leakage:  all features shifted by 1 year before scoring
Target:        next-year EV growth (same as EVQTI)
Normalisation: cross-section rank-transform per year → z-score
Weights:       empirical |IC| calibration (same methodology as EVQTI v3)

Output:
  - composite_score  (0-100, higher = more attractive early-stage opportunity)
  - factor_scores    (per-factor contribution per country, for explanation)
  - ic_by_factor     (empirical IC per factor, for validation)
"""


import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Factor definitions:
# (column_in_panel, direction, group)
# direction: -1 = lower value = more attractive (early-stage)
#            +1 = higher value = more attractive
_FACTORS = {
    "penetration_stock":  ("ev_stock_share",    -1, "Penetration"),
    "penetration_sales":  ("ev_sales_share",    -1, "Penetration"),
    "infra_gap":          ("ev_per_charger",     +1, "Infrastructure"),
    "energy_cost":        ("gasoline_usd",       +1, "Energy"),
    "macro_gdp":          ("gdp",                +1, "Macro"),
    "momentum":           ("momentum_3y",        +1, "Momentum"),
}

_WIN_LO, _WIN_HI   = 0.01, 0.99
_TARGET_WIN_LO, _TARGET_WIN_HI = 0.05, 0.95
_MIN_COUNTRIES = 10


class FactorEngine:
    """
    Multi-factor EV Transition scoring engine.

    Backward compatible: get_evqti_scores() returns the same output
    as EVQTI.get_ranking() for any code that depends on EVQTI directly.

    Usage
    -----
    engine = FactorEngine()
    engine.fit(df, fuel_df=fuel_df)
    ranking = engine.get_ranking()
    factors = engine.get_factor_contributions()
    summary = engine.get_factor_ic()
    narrative = engine.get_drivers(country="Brazil")
    """

    def __init__(self, weights: dict | None = None):
        self.weights      = weights or {}   # override calibrated weights if needed
        self.fitted       = False
        self.panel_       = pd.DataFrame()
        self.scores_      = pd.DataFrame()
        self.ic_by_factor_= pd.DataFrame()
        self._calibrated_weights: dict = {}
        self.calibration_end_year_: int | None = None
        self.as_of_year_: int | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        fuel_df: pd.DataFrame | None = None,
        infra_df: pd.DataFrame | None = None,
        calibration_end_year: int | None = None,
    ) -> "FactorEngine":
        """
        Build the panel, calibrate weights using labels observable by the scoring origin,
        and score the latest panel year.

        Temporal contract
        -----------------
        ``as_of_year`` is the latest year present in ``df`` and is the forecast origin.
        ``calibration_end_year`` is the last *panel row year* admitted to calibration,
        not the forecast origin. A row at year t carries ``growth_t1`` from
        ``growth_rate`` at t+1. Therefore rows through as_of_year - 1 have labels
        observable by the as_of_year origin. The default cutoff is as_of_year - 1.

        Parameters
        ----------
        df       : ETIL processed panel (output of FeatureEngineer)
        fuel_df  : Optional IEA fuel price DataFrame (date, country, gasoline_usd)
        infra_df : Optional ev_per_charger DataFrame (country, year, ev_per_charger)
                   from IEALoader.get_aggregator(2026).ev_per_charger()
                   If None, uses the panel's ev_per_charger column (may be unreliable).
        """
        if "year" not in df.columns or df["year"].dropna().empty:
            logger.error("FactorEngine: input has no usable year column.")
            return self
        as_of_year = int(pd.to_numeric(df["year"], errors="coerce").max())
        cutoff = as_of_year - 1 if calibration_end_year is None else int(calibration_end_year)
        if cutoff >= as_of_year:
            raise ValueError("calibration_end_year must be before the latest scoring year.")
        self.as_of_year_ = as_of_year
        self.calibration_end_year_ = cutoff

        panel = self._build_panel(df, fuel_df, infra_df, calibration_end_year=cutoff)
        if panel.empty:
            logger.error("FactorEngine: empty panel after build.")
            return self

        self._calibrated_weights = self.weights or self._calibrate_weights(panel, cutoff)
        self.scores_  = self._score_panel(panel)
        self.panel_   = panel
        self.fitted   = True

        n = self.scores_["country"].nunique()
        logger.info(
            f"FactorEngine fitted | {n} countries | "
            f"factors: {list(self._calibrated_weights.keys())}"
        )
        return self

    def walk_forward(
        self,
        df: pd.DataFrame,
        min_train_years: int = 6,
        step: int = 1,
        fuel_df: pd.DataFrame | None = None,
        infra_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Expanding-origin evaluation of the ranking signal.

        For each origin year T, the model receives data only through T.
        Its calibration cutoff is therefore T-1: row T-1's growth_t1 is
        growth_rate[T], which is observable at origin T. The score is produced
        for row T using its lagged (T-1) features. The evaluation target,
        growth_rate[T+1], is joined only after scoring and is never passed to fit.

        Returns one row per country and origin, including the raw, un-winsorized
        evaluation target. This evaluates rank association, not a point forecast.
        """
        if min_train_years < 2:
            raise ValueError("min_train_years must be at least 2.")
        if step < 1:
            raise ValueError("step must be at least 1.")
        if "year" not in df.columns or "country" not in df.columns:
            raise ValueError("df must contain country and year columns.")

        source = df.copy()
        source["year"] = pd.to_numeric(source["year"], errors="coerce")
        source = source.dropna(subset=["year", "country"]).copy()
        source["year"] = source["year"].astype(int)
        if "growth_rate" not in source.columns:
            if "ev_sales" not in source.columns:
                raise ValueError("df must contain growth_rate or ev_sales.")
            source = source.sort_values(["country", "year"])
            source["growth_rate"] = (
                source.groupby("country")["ev_sales"].pct_change() * 100.0
            )

        years = sorted(source["year"].unique())
        records = []
        for position, origin_year in enumerate(years):
            if position < min_train_years or (position - min_train_years) % step:
                continue

            # Hard boundary: no source rows after this origin enter model.fit().
            origin_data = source.loc[source["year"].le(origin_year)].copy()
            model = FactorEngine(weights=self.weights)
            model.fit(origin_data, fuel_df=fuel_df, infra_df=infra_df)
            if not model.fitted:
                continue

            score_panel = model.panel_.loc[
                model.panel_["year"].eq(origin_year)
            ].copy()
            if score_panel.empty:
                continue
            scores = model._score_panel(score_panel)
            if scores.empty:
                continue

            # Future outcome is joined after the model has produced its score.
            ordered = source.sort_values(["country", "year"]).copy()
            ordered["_evaluation_target"] = (
                ordered.groupby("country")["growth_rate"].shift(-1)
            )
            targets = ordered.loc[
                ordered["year"].eq(origin_year),
                ["country", "_evaluation_target"],
            ]
            evaluated = scores.merge(targets, on="country", how="left")
            if not evaluated["_evaluation_target"].notna().any():
                continue
            valid = evaluated.dropna(subset=["composite_score", "_evaluation_target"])
            if len(valid) >= _MIN_COUNTRIES:
                ic, p_value = stats.spearmanr(
                    valid["composite_score"], valid["_evaluation_target"]
                )
            else:
                ic, p_value = np.nan, np.nan

            for _, row in evaluated.iterrows():
                records.append({
                    "origin_year": origin_year,
                    "score_year": origin_year,
                    "country": row["country"],
                    "composite_score": row["composite_score"],
                    "target_growth_t1": row["_evaluation_target"],
                    "spearman_ic": float(ic) if not np.isnan(ic) else np.nan,
                    "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                    "n_evaluated": int(len(valid)),
                    "calibration_end_year": model.calibration_end_year_,
                    "n_train_years": int(position),
                })

        return pd.DataFrame(records)

    def get_ranking(self, n: int | None = None) -> pd.DataFrame:
        """
        Full country ranking by composite score.

        Returns
        -------
        pd.DataFrame
            country, rank, composite_score, evqti_compat_score, n_factors_used
        """
        self._check_fitted()
        df = self.scores_.copy()
        df["rank"] = df["composite_score"].rank(ascending=False).astype(int)
        df = df.sort_values("rank")
        return df.head(n) if n else df

    def get_factor_contributions(self) -> pd.DataFrame:
        """
        Per-country, per-factor score contribution and direction.

        Returns
        -------
        pd.DataFrame
            country, factor, factor_group, z_score, contribution, weight, direction
        """
        self._check_fitted()
        if "factor_breakdown" not in self.panel_.columns:
            return pd.DataFrame()
        return self._factor_breakdown_

    def get_factor_ic(self) -> pd.DataFrame:
        """
        In-sample calibration IC (Spearman correlation with next-year growth).

        This is a calibration diagnostic, not out-of-sample validation. Use
        walk_forward() for expanding-origin rank-association evaluation.
        """
        self._check_fitted()
        return self.ic_by_factor_.copy()

    def get_drivers(self, country: str) -> dict:
        """
        Top drivers (positive and negative) for a specific country.

        Returns a dict with:
          - top_positive: factors pushing the score up
          - top_negative: factors pulling the score down
          - regime_compatible: whether score aligns with current momentum
          - headline: one-line summary
        """
        self._check_fitted()
        if not hasattr(self, "_factor_breakdown_") or self._factor_breakdown_.empty:
            return {}

        country_factors = self._factor_breakdown_[
            self._factor_breakdown_["country"] == country
        ].copy()
        if country_factors.empty:
            return {"headline": f"No factor data available for {country}"}

        country_factors = country_factors.sort_values("contribution", ascending=False)
        score_row = self.scores_[self.scores_["country"] == country]
        score = float(score_row["composite_score"].iloc[0]) if not score_row.empty else None

        # Compute rank on the fly from scores_ (rank not stored, computed in get_ranking)
        all_scores = self.scores_.set_index("country")["composite_score"]
        rank = int((all_scores > (score or 0)).sum() + 1) if score is not None else None

        top_pos = country_factors[country_factors["contribution"] > 0].head(3)
        top_neg = country_factors[country_factors["contribution"] < 0].head(2)

        # Build headline
        pos_labels = " + ".join(top_pos["factor"].str.replace("_", " ").str.title().tolist())
        headline = (
            f"#{rank} globally — driven by {pos_labels}" if pos_labels
            else f"#{rank} globally"
        )

        return {
            "country":       country,
            "rank":          rank,
            "score":         score,
            "headline":      headline,
            "top_positive":  top_pos[["factor","factor_group","contribution","weight"]].to_dict("records"),
            "top_negative":  top_neg[["factor","factor_group","contribution","weight"]].to_dict("records"),
            "n_factors":     int(country_factors["weight"].astype(bool).sum()),
        }

    def get_evqti_scores(self) -> pd.DataFrame:
        """
        Backward-compatible output matching EVQTI.get_ranking() format.
        Uses penetration factors only, same as EVQTI v3.
        """
        self._check_fitted()
        scores = self.scores_[["country", "evqti_compat_score"]].copy()
        scores = scores.rename(columns={"evqti_compat_score": "evqti_score"})
        scores["rank"] = scores["evqti_score"].rank(ascending=False, method="min").astype(int)
        return scores[["country", "rank", "evqti_score"]].sort_values(["rank", "country"])

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_panel(
        self,
        df: pd.DataFrame,
        fuel_df: pd.DataFrame | None,
        infra_df: pd.DataFrame | None = None,
        calibration_end_year: int | None = None,
    ) -> pd.DataFrame:
        """
        Merge all feature sources into a single lagged panel.
        All features are shifted 1 year to ensure t-1 alignment.
        """
        panel = df.copy()
        panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
        if calibration_end_year is None:
            calibration_end_year = int(panel["year"].max()) - 1

        # Merge IEA ev_per_charger (real values from Aggregator, not derived column)
        if infra_df is not None and not infra_df.empty:
            try:
                infra_clean = infra_df[["country","year","ev_per_charger"]].copy()
                panel = panel.drop(columns=["ev_per_charger"], errors="ignore")
                panel = panel.merge(infra_clean, on=["country","year"], how="left")
                logger.info(f"Infra data merged: {infra_clean['country'].nunique()} countries")
            except Exception as e:
                logger.warning(f"Infra merge failed: {e}")

        # Merge fuel prices
        if fuel_df is not None and not fuel_df.empty:
            try:
                fuel_annual = (
                    fuel_df.copy()
                    .assign(year=lambda d: pd.to_datetime(d["date"]).dt.year)
                    .groupby(["country","year"])["gasoline_usd"]
                    .mean()
                    .reset_index()
                )
                panel = panel.merge(fuel_annual, on=["country","year"], how="left")
            except Exception as e:
                logger.warning(f"Fuel merge failed: {e}. Continuing without fuel factor.")

        # Create lagged features for all factors
        active_factors = {}
        for factor_name, (col, direction, group) in _FACTORS.items():
            if col not in panel.columns:
                logger.debug(f"Factor '{factor_name}' skipped: column '{col}' not in panel.")
                continue
            lag_col = f"{col}_lag1"
            panel[lag_col] = (
                panel.sort_values(["country", "year"])
                .groupby("country")[col]
                .transform(lambda x: x.shift(1))
            )
            # Winsorise using calibration years only; never use future rows.
            calibration_values = panel.loc[
                panel["year"].le(calibration_end_year), lag_col
            ].dropna()
            if len(calibration_values) >= 10:
                lo = calibration_values.quantile(_WIN_LO)
                hi = calibration_values.quantile(_WIN_HI)
                panel[lag_col] = panel[lag_col].clip(lo, hi)
            active_factors[factor_name] = (lag_col, direction, group)

        # Compute target: next-year growth (t+1)
        panel["growth_t1"] = (
            panel.sort_values(["country", "year"])
            .groupby("country")["growth_rate"]
            .transform(lambda x: x.shift(-1))
        )
        # A row t's target is growth_rate[t+1]. At forecast origin as_of_year,
        # the last eligible calibration row is as_of_year - 1 (default cutoff),
        # because its target is the observed growth_rate[as_of_year].
        # The explicit cutoff is a panel-row cutoff, not an as-of date.
        calibration_target = panel.loc[
            panel["year"].le(calibration_end_year), "growth_t1"
        ].dropna()
        if len(calibration_target) >= 10:
            lo_t = calibration_target.quantile(_TARGET_WIN_LO)
            hi_t = calibration_target.quantile(_TARGET_WIN_HI)
            panel["growth_t1_win"] = panel["growth_t1"].clip(lo_t, hi_t)
        else:
            panel["growth_t1_win"] = panel["growth_t1"]

        self._active_factors = active_factors
        return panel

    def _calibrate_weights(self, panel: pd.DataFrame, calibration_end_year: int) -> dict:
        """
        Calibrate factor weights by empirical |IC| against next-year growth.
        Factors with |IC| < 0.05 or p > 0.20 are excluded.
        """
        ic_rows = []
        for factor_name, (lag_col, direction, group) in self._active_factors.items():
            year_ics = []
            # Eligible labels are those whose t+1 outcome is observable by
            # the fit's as_of_year_. fit() guarantees calibration_end_year < as_of_year.
            for yr in sorted(panel.loc[panel["year"].le(calibration_end_year), "year"].unique()):
                sub = panel[panel["year"] == yr][
                    [lag_col, "growth_t1_win"]
                ].dropna()
                if len(sub) < _MIN_COUNTRIES:
                    continue
                ic, pval = stats.spearmanr(sub[lag_col] * direction, sub["growth_t1_win"])
                if not np.isnan(ic):
                    year_ics.append({"year": yr, "ic": ic, "p_value": pval})

            if not year_ics:
                continue

            ic_df  = pd.DataFrame(year_ics)
            mean_ic = ic_df["ic"].mean()
            pct_sig = (ic_df["p_value"] < 0.10).mean()

            # Include if |IC| >= 0.05 and at least 20% years significant
            if abs(mean_ic) >= 0.05 and pct_sig >= 0.20:
                ic_rows.append({
                    "factor":       factor_name,
                    "factor_group": self._active_factors[factor_name][2],
                    "mean_ic":      round(mean_ic, 3),
                    "abs_ic":       round(abs(mean_ic), 3),
                    "pct_sig":      round(pct_sig, 2),
                    "lag_col":      lag_col,
                    "direction":    direction,
                })

        self.ic_by_factor_ = pd.DataFrame(ic_rows)

        if self.ic_by_factor_.empty:
            logger.warning("No factors passed IC threshold — falling back to penetration only.")
            fallback = {
                "penetration_stock": 0.485,
                "penetration_sales": 0.515,
            }
            return fallback

        # Normalise |IC| to sum to 1
        total_abs_ic = self.ic_by_factor_["abs_ic"].sum()
        weights = {}
        for _, row in self.ic_by_factor_.iterrows():
            weights[row["factor"]] = row["abs_ic"] / total_abs_ic

        logger.info(f"Factor weights calibrated: {weights}")
        return weights

    def _score_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Score each country using the calibrated multi-factor weights.
        Z-score normalise cross-section per year, then weight and aggregate.
        """
        breakdown_rows = []

        # Score using latest year per country
        latest_per_country = (
            panel.dropna(subset=["ev_sales"])
            .groupby("country")["year"]
            .max()
            .reset_index()
        )
        latest_panel = panel.merge(latest_per_country, on=["country","year"])

        # Cross-section z-score per year for each factor
        for year in latest_panel["year"].unique():
            yr_mask = latest_panel["year"] == year
            for factor_name, weight in self._calibrated_weights.items():
                if factor_name not in self._active_factors:
                    continue
                lag_col, direction, group = self._active_factors[factor_name]
                if lag_col not in latest_panel.columns:
                    continue
                yr_vals = latest_panel.loc[yr_mask, lag_col].copy()
                z = (yr_vals - yr_vals.mean()) / (yr_vals.std() + 1e-9)
                z_directed = z * direction
                contribution = z_directed * weight
                for idx in latest_panel[yr_mask].index:
                    country = latest_panel.loc[idx, "country"]
                    breakdown_rows.append({
                        "country":      country,
                        "year":         year,
                        "factor":       factor_name,
                        "factor_group": group,
                        "z_score":      float(z_directed.get(idx, np.nan)),
                        "contribution": float(contribution.get(idx, np.nan)),
                        "weight":       weight,
                        "direction":    direction,
                    })

        breakdown = pd.DataFrame(breakdown_rows)
        self._factor_breakdown_ = breakdown

        # Aggregate: composite score = sum of weighted z-scores → normalize to 0-100
        composite = (
            breakdown.groupby("country")["contribution"]
            .sum()
            .reset_index()
            .rename(columns={"contribution": "composite_raw"})
        )
        lo = composite["composite_raw"].min()
        hi = composite["composite_raw"].max()
        composite["composite_score"] = ((composite["composite_raw"] - lo) / (hi - lo + 1e-9)) * 100
        composite["composite_score"] = composite["composite_score"].round(2)
        composite["n_factors_used"]  = len(self._calibrated_weights)

        # Evqti_compat_score: penetration factors only (for backward compatibility)
        pen_factors = [f for f in self._calibrated_weights if "penetration" in f]
        if pen_factors:
            pen_contrib = (
                breakdown[breakdown["factor"].isin(pen_factors)]
                .groupby("country")["contribution"]
                .sum()
                .reset_index()
                .rename(columns={"contribution": "evqti_raw"})
            )
            lo_p = pen_contrib["evqti_raw"].min()
            hi_p = pen_contrib["evqti_raw"].max()
            pen_contrib["evqti_compat_score"] = (
                (pen_contrib["evqti_raw"] - lo_p) / (hi_p - lo_p + 1e-9) * 100
            ).round(2)
            composite = composite.merge(pen_contrib[["country","evqti_compat_score"]], on="country", how="left")
        else:
            composite["evqti_compat_score"] = composite["composite_score"]

        return composite

    def _check_fitted(self):
        if not self.fitted:
            raise ValueError("FactorEngine not fitted. Call fit() first.")
