# models/scoring/evqti.py
"""
EVQTI — EV Quant Transition Index (v3)

Fator de pesquisa com pesos de referência fixos; o backtest recalibra pesos somente com labels anteriores.

LÓGICA ECONÔMICA:
  Lei de difusão de Bass aplicada ao cross-section de países:
  mercados com baixa penetração atual concentram o maior crescimento futuro.

FEATURES (somente as com IC estatisticamente significativo p<0.05):
  - ev_stock_share_lag1  : IC = -0.259  peso = 0.485 (invertido: early-stage)
  - ev_sales_share_lag1  : IC = -0.275  peso = 0.515 (invertido: early-stage)

  growth_rate_lag1 foi removida: IC = -0.037, p = 0.36 (não significativo).
  Momentum de 1 ano é ruído no cross-section de países; não prediz crescimento futuro.

NORMALIZAÇÃO:
  1. Winsorização 1%-99% nas features RAW (antes de criar lags)
  2. Winsorização 5%-95% no target growth_t+1
  3. Z-score cross-section
  4. Score = -(w1*z_stock + w2*z_sales)  — sinal negativo = early-stage positivo

VALIDAÇÃO:
  - IC (Pearson sobre ranks) e Rank IC (Spearman) por ano
  - Análise de quintis com IC por quintil
  - Information Ratio = IC_médio / std(IC)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Pesos de referência derivados do IC empírico no painel histórico IEA 2026
# (Cars, 2010-2025). São pesos descritivos de referência, NÃO pesos point-in-time.
# O backtest abaixo estima novos pesos apenas com os anos de treino de cada origem.
# |IC| stock=0.259 / (0.259+0.275) = 0.4851
# |IC| sales=0.275 / (0.259+0.275) = 0.5149
_WEIGHTS = {
    "ev_stock_share": (-1.0, 0.485),  # (sinal, peso) — invertido: early-stage
    "ev_sales_share": (-1.0, 0.515),  # invertido: early-stage
    # growth_rate EXCLUÍDA: IC=-0.037, p=0.36 (não significativo)
}

# Winsorização do target: 5%-95% (mais conservador que 1%-99%)
_TARGET_WIN_LOW  = 0.05
_TARGET_WIN_HIGH = 0.95


class EVQTI:
    """
    Fator quantitativo de transição EV — versão 3 com pesos calibrados por IC.
    """

    def __init__(self, weights: Optional[dict] = None):
        self.weights        = weights or _WEIGHTS
        self.fitted         = False
        self.country_scores: Optional[pd.Series]   = None
        self.panel_:         Optional[pd.DataFrame] = None
        self.ic_stats_:      Optional[pd.DataFrame] = None
        self.decile_stats_:  Optional[pd.DataFrame] = None
        self._ic_mean:       float = 0.0
        self._ir:            float = 0.0

    # ── API pública ──────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "EVQTI":
        panel = self._build_panel(df)
        self.panel_ = panel

        latest = int(panel["year"].max())
        df_score = self._compute_score(panel[panel["year"] == latest].copy())
        self.country_scores = (
            df_score.set_index("country")["evqti_score"]
            .sort_values(ascending=False)
        )

        self.ic_stats_     = self._compute_ic(panel)
        self.decile_stats_ = self._compute_deciles(panel)

        if not self.ic_stats_.empty:
            self._ic_mean = self.ic_stats_["rank_ic"].mean()
            self._ir      = self._ic_mean / (self.ic_stats_["rank_ic"].std() + 1e-9)

        self.fitted = True
        logger.info(
            f"EVQTI v3 | {len(self.country_scores)} países | "
            f"IC={self._ic_mean:.3f} | IR={self._ir:.2f}"
        )
        return self

    def get_ranking(self) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Chame fit() primeiro.")
        rdf = self.country_scores.reset_index()
        rdf.columns = ["country", "evqti_score"]
        rdf["rank"] = np.arange(1, len(rdf) + 1)
        return rdf[["country", "rank", "evqti_score"]]

    def get_ic_stats(self) -> pd.DataFrame:
        if self.ic_stats_ is None:
            return pd.DataFrame()
        return self.ic_stats_

    def get_decile_stats(self) -> pd.DataFrame:
        if self.decile_stats_ is None:
            return pd.DataFrame()
        return self.decile_stats_

    def get_summary(self) -> dict:
        """Retorna métricas resumidas do fator."""
        if not self.fitted or self.ic_stats_.empty:
            return {}
        ic = self.ic_stats_["rank_ic"]
        dec = self.decile_stats_
        spread = (
            dec["avg_growth"].iloc[-1] - dec["avg_growth"].iloc[0]
            if len(dec) >= 2 else np.nan
        )
        return {
            "ic_mean":       round(float(ic.mean()), 3),
            "ic_std":        round(float(ic.std()), 3),
            "ir":            round(float(self._ir), 3),
            "pct_positive":  round(float((ic > 0).mean() * 100), 1),
            "pct_sig":       round(float(self.ic_stats_["significant"].mean() * 100), 1),
            "spread_q5_q1":  round(float(spread), 1) if not np.isnan(spread) else None,
            "n_countries":   len(self.country_scores),
        }

    def backtest(
        self,
        df: pd.DataFrame,
        train_years: int = 6,
        test_years: int = 1,
        step: int = 1,
        min_countries: int = 10,
    ) -> pd.DataFrame:
        """
        Rolling-origin backtest with weights calibrated on past labels only.

        At test origin T, calibration uses rows t < T, whose t+1 outcomes
        are observable by T. The test score uses lagged features at T and
        is evaluated against raw growth_t1 (not the winsorized diagnostic).

        Annual IEA revisions/publication vintages are not available, so this
        is temporal-order validation, not a fully vintage-aware real-time test.
        """
        if train_years < 3:
            raise ValueError("train_years must be at least 3.")
        if test_years < 1 or step < 1 or min_countries < 3:
            raise ValueError("test_years, step and min_countries must be positive.")

        panel = self._build_panel(df)
        years = sorted(int(y) for y in panel["year"].dropna().unique())
        year_set = set(years)
        feature_directions = {
            "ev_stock_share": -1.0,
            "ev_sales_share": -1.0,
        }
        yearly_results = []
        min_calibration_years = max(3, (train_years + 1) // 2)

        for test_year in years:
            calibration_years = list(range(test_year - train_years, test_year))
            if not all(year in year_set for year in calibration_years):
                continue

            calibrated_ic = {}
            calibration_counts = {}
            for feature, direction in feature_directions.items():
                lag_col = f"{feature}_lag1"
                annual_ics = []
                for train_year in calibration_years:
                    sample = panel.loc[
                        panel["year"].eq(train_year), [lag_col, "growth_t1"]
                    ].dropna()
                    if (
                        len(sample) < min_countries
                        or sample[lag_col].nunique() < 2
                        or sample["growth_t1"].nunique() < 2
                    ):
                        continue
                    ic, _ = stats.spearmanr(sample[lag_col], sample["growth_t1"])
                    if np.isfinite(ic):
                        annual_ics.append(float(direction * ic))

                calibration_counts[feature] = len(annual_ics)
                if len(annual_ics) >= min_calibration_years:
                    calibrated_ic[feature] = float(np.mean(annual_ics))

            if not calibrated_ic:
                continue

            # Only positive train-period directional IC contributes to weight.
            raw_weights = {
                feature: max(value, 0.0)
                for feature, value in calibrated_ic.items()
            }
            weight_total = sum(raw_weights.values())
            if weight_total > 1e-12:
                weights = {
                    feature: value / weight_total
                    for feature, value in raw_weights.items()
                }
                calibration_status = "train_only_ic"
            else:
                # Explicit neutral fallback when training data supports no signal.
                weights = {
                    feature: 1.0 / len(calibrated_ic)
                    for feature in calibrated_ic
                }
                calibration_status = "equal_weight_fallback"

            test = panel.loc[panel["year"].eq(test_year)].copy()
            components = []
            used_weights = {}
            for feature, weight in weights.items():
                lag_col = f"{feature}_lag1"
                if lag_col not in test.columns:
                    continue
                values = test[lag_col]
                if values.notna().mean() < 0.20:
                    continue
                median = values.median()
                if pd.isna(median):
                    continue
                filled = values.fillna(median)
                sd = filled.std()
                if not np.isfinite(sd) or sd < 1e-9:
                    continue
                z_score = (filled - filled.mean()) / sd * feature_directions[feature]
                components.append((z_score, weight))
                used_weights[feature] = weight

            used_weight_total = sum(used_weights.values())
            if not components or used_weight_total <= 1e-12:
                continue

            score = sum(z * weight for z, weight in components) / used_weight_total
            evaluation = pd.DataFrame({
                "country": test["country"],
                "score": score,
                "target": test["growth_t1"],
            }).dropna()

            if (
                len(evaluation) < min_countries
                or evaluation["score"].nunique() < 2
                or evaluation["target"].nunique() < 2
            ):
                continue

            rank_ic, rank_p = stats.spearmanr(
                evaluation["score"], evaluation["target"]
            )
            pearson_ic, _ = stats.pearsonr(
                evaluation["score"], evaluation["target"]
            )
            if not np.isfinite(rank_ic):
                continue

            # Stable country ordering makes quintile assignment deterministic
            # when multiple countries have identical scores.
            evaluation = evaluation.sort_values("country").copy()
            score_rank = evaluation["score"].rank(method="first")
            evaluation["quintile"] = pd.qcut(
                score_rank, q=5, labels=False, duplicates="drop"
            )
            q_means = evaluation.groupby("quintile")["target"].mean()
            spread = (
                float(q_means.iloc[-1] - q_means.iloc[0])
                if len(q_means) >= 2 else np.nan
            )

            yearly_results.append({
                "test_year": test_year,
                "window_start": test_year,
                "window_end": test_year,
                "ic_mean": float(rank_ic),
                "ic_std": 0.0,
                "rank_ic_pvalue": float(rank_p),
                "pearson_ic": float(pearson_ic),
                "top_bottom_spread": spread,
                "n_years": 1,
                "n_obs": int(len(evaluation)),
                "calibration_start_year": calibration_years[0],
                "calibration_end_year": test_year - 1,
                "calibration_years_stock": calibration_counts.get("ev_stock_share", 0),
                "calibration_years_sales": calibration_counts.get("ev_sales_share", 0),
                "mean_train_ic_stock": calibrated_ic.get("ev_stock_share", np.nan),
                "mean_train_ic_sales": calibrated_ic.get("ev_sales_share", np.nan),
                "weight_stock": used_weights.get("ev_stock_share", 0.0) / used_weight_total,
                "weight_sales": used_weights.get("ev_sales_share", 0.0) / used_weight_total,
                "calibration_status": calibration_status,
            })

        if not yearly_results:
            return pd.DataFrame()

        annual = pd.DataFrame(yearly_results).sort_values("test_year").reset_index(drop=True)
        windows = []
        for start_idx in range(0, len(annual) - test_years + 1, step):
            block = annual.iloc[start_idx:start_idx + test_years]
            windows.append({
                "window_start": int(block["test_year"].iloc[0]),
                "window_end": int(block["test_year"].iloc[-1]),
                "ic_mean": float(block["ic_mean"].mean()),
                "ic_std": float(block["ic_mean"].std(ddof=0)),
                "n_years": int(len(block)),
                "n_obs": int(block["n_obs"].sum()),
                "rank_ic_pvalue": float(block["rank_ic_pvalue"].mean()),
                "pearson_ic": float(block["pearson_ic"].mean()),
                "top_bottom_spread": float(block["top_bottom_spread"].mean()),
                "calibration_start_year": int(block["calibration_start_year"].min()),
                "calibration_end_year": int(block["calibration_end_year"].max()),
                "weight_stock": float(block["weight_stock"].mean()),
                "weight_sales": float(block["weight_sales"].mean()),
                "calibration_status": ";".join(sorted(block["calibration_status"].unique())),
            })
        return pd.DataFrame(windows)

    # ── Construção do painel ─────────────────────────────────────────────

    def _build_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build lagged features and next-year target with expanding, point-in-time
        winsorization.

        For a score row at year t:
        - lagged features are clipped using only lag observations from rows < t;
        - growth_t1 is an evaluation label, not a scoring input;
        - target clipping bounds use only target labels from rows < t, whose
          t+1 outcomes are observable by the end of year t.

        This prevents future years from changing historical preprocessing.
        """
        panel = df.sort_values(["country", "year"]).copy()
        panel["year"] = pd.to_numeric(panel["year"], errors="coerce")

        if "ev_sales_share" not in panel.columns:
            if "total_sales" in panel.columns and panel["total_sales"].notna().any():
                panel["ev_sales_share"] = (
                    panel["ev_sales"] / panel["total_sales"].replace(0, np.nan) * 100
                )
            else:
                panel["ev_sales_share"] = np.nan

        if "growth_rate" not in panel.columns:
            previous_sales = panel[["country", "year", "ev_sales"]].copy()
            previous_sales["year"] = previous_sales["year"] + 1
            previous_sales = previous_sales.rename(
                columns={"ev_sales": "_ev_sales_previous_year"}
            )
            panel = panel.merge(
                previous_sales[["country", "year", "_ev_sales_previous_year"]],
                on=["country", "year"],
                how="left",
                validate="one_to_one",
            )
            denominator = panel["_ev_sales_previous_year"].replace(0, np.nan)
            panel["growth_rate"] = (
                panel["ev_sales"] / denominator - 1
            ) * 100
            panel = panel.drop(columns=["_ev_sales_previous_year"])

        years = sorted(panel["year"].dropna().unique())

        # Create raw lags first. Bounds are fitted separately at each year
        # from earlier rows only, then applied to that year's cross-section.
        for col in ["ev_stock_share", "ev_sales_share"]:
            if col not in panel.columns:
                continue
            lag_col = f"{col}_lag1"
            lag_source = panel[["country", "year", col]].copy()
            lag_source["year"] = lag_source["year"] + 1
            lag_source = lag_source.rename(columns={col: lag_col})
            panel = panel.drop(columns=[lag_col], errors="ignore").merge(
                lag_source[["country", "year", lag_col]],
                on=["country", "year"],
                how="left",
                validate="one_to_one",
            )
            raw_lag = panel[lag_col].copy()
            clipped_lag = raw_lag.copy()

            for year in years:
                current = panel["year"].eq(year)
                history = raw_lag.loc[panel["year"].lt(year)].dropna()
                if len(history) < 10:
                    continue
                lo = history.quantile(0.01)
                hi = history.quantile(0.99)
                clipped_lag.loc[current] = raw_lag.loc[current].clip(lo, hi)

            panel[lag_col] = clipped_lag

        # Target is retained raw for evaluation; its clipped companion is
        # diagnostic only and never enters the score calculation.
        target_source = panel[["country", "year", "growth_rate"]].copy()
        target_source["year"] = target_source["year"] - 1
        target_source = target_source.rename(columns={"growth_rate": "growth_t1"})
        panel = panel.drop(columns=["growth_t1"], errors="ignore").merge(
            target_source[["country", "year", "growth_t1"]],
            on=["country", "year"],
            how="left",
            validate="one_to_one",
        )
        raw_target = panel["growth_t1"].copy()
        clipped_target = raw_target.copy()

        for year in years:
            current = panel["year"].eq(year)
            # Row r<year has target growth_rate[r+1], known by origin=year.
            history = raw_target.loc[panel["year"].lt(year)].dropna()
            if len(history) < 10:
                continue
            lo = history.quantile(_TARGET_WIN_LOW)
            hi = history.quantile(_TARGET_WIN_HIGH)
            clipped_target.loc[current] = raw_target.loc[current].clip(lo, hi)

        panel["growth_t1_win"] = clipped_target
        return panel

    def _compute_score(self, df_cross: pd.DataFrame) -> pd.DataFrame:
        """
        Score cross-section com z-score e pesos calibrados.
        Apenas features com cobertura >= 20% são incluídas.
        """
        df_cross = df_cross.copy()
        components  = []
        weight_sum  = 0.0

        for feat, (sign, weight) in self.weights.items():
            col = f"{feat}_lag1"
            if col not in df_cross.columns:
                col = feat
            if col not in df_cross.columns:
                continue

            valid_frac = df_cross[col].notna().mean()
            if valid_frac < 0.20:
                logger.warning(f"Feature {feat}: cobertura {valid_frac:.0%} < 20% — ignorada.")
                continue

            sub = df_cross[col].fillna(df_cross[col].median())
            mu  = sub.mean()
            sd  = sub.std()
            if sd < 1e-9:
                continue

            z = (sub - mu) / sd * sign
            components.append(z * weight)
            weight_sum += weight

        if not components:
            raise ValueError("Nenhuma feature válida para EVQTI.")

        df_cross["evqti_score"] = sum(components) / weight_sum
        return df_cross

    # ── Validação estatística ────────────────────────────────────────────

    def _compute_ic(self, panel: pd.DataFrame) -> pd.DataFrame:
        """IC e Rank IC por ano com t-stat."""
        years = sorted(panel["year"].unique())[1:-1]
        rows  = []

        for yr in years:
            df_yr = self._compute_score(panel[panel["year"] == yr].copy())
            sub   = df_yr[["country", "evqti_score", "growth_t1_win"]].dropna()
            if len(sub) < 10:
                continue

            ic_p, _  = stats.pearsonr(sub["evqti_score"],        sub["growth_t1_win"])
            ic_s, pv = stats.spearmanr(sub["evqti_score"],        sub["growth_t1_win"])

            # t-stat do IC: t = IC * sqrt(n-2) / sqrt(1-IC²)
            n      = len(sub)
            t_stat = ic_s * np.sqrt(n - 2) / np.sqrt(max(1 - ic_s**2, 1e-9))

            rows.append({
                "year":        yr,
                "ic":          round(ic_p,  4),
                "rank_ic":     round(ic_s,  4),
                "t_stat":      round(t_stat, 3),
                "p_value":     round(pv,    4),
                "n":           n,
                "significant": pv < 0.05,
            })

        return pd.DataFrame(rows)

    def _compute_deciles(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Quintis pooled — crescimento médio futuro por quintil de EVQTI."""
        years    = sorted(panel["year"].unique())[1:-1]
        all_rows = []

        for yr in years:
            df_yr = self._compute_score(panel[panel["year"] == yr].copy())
            sub   = df_yr[["country", "evqti_score", "growth_t1_win"]].dropna()
            if len(sub) < 15:
                continue
            try:
                sub = sub.copy()
                sub["decile"] = pd.qcut(sub["evqti_score"], 5, labels=False, duplicates="drop")
                sub["year"]   = yr
                all_rows.append(sub)
            except Exception:
                pass

        if not all_rows:
            return pd.DataFrame(columns=["decile", "avg_growth", "std_growth", "n", "decile_label"])

        full = pd.concat(all_rows, ignore_index=True)
        stats_df = (
            full.groupby("decile")["growth_t1_win"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_growth", "std": "std_growth", "count": "n"})
        )
        stats_df["se"] = stats_df["std_growth"] / np.sqrt(stats_df["n"])
        stats_df["decile_label"] = stats_df["decile"].map({
            0: "Q1 — Baixo",
            1: "Q2",
            2: "Q3",
            3: "Q4",
            4: "Q5 — Alto",
        })
        return stats_df
