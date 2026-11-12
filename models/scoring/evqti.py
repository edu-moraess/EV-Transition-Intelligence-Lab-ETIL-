# models/scoring/evqti.py
"""
EVQTI — EV Quant Transition Index (v3)

Fator quantitativo anti-leakage com pesos calibrados empiricamente via IC.

LÓGICA ECONÔMICA:
  Lei de difusão de Bass aplicada ao cross-section de países:
  mercados com baixa penetração atual concentram o maior crescimento futuro.

FEATURES (somente as com IC estatisticamente significativo p<0.05):
  - ev_stock_share_lag1  : IC = -0.259  peso = 0.485 (invertido: early-stage)
  - ev_sales_share_lag1  : IC = -0.275  peso = 0.515 (invertido: early-stage)

  growth_rate_lag1 foi removida: IC = -0.037, p = 0.36 (não significativo).
  Momentum de 1 ano é ruído no cross-section de países; não prediz crescimento futuro.

NORMALIZAÇÃO:
  1. Criar lags de calendário exatos (t-1); anos ausentes permanecem missing
  2. Winsorização expansiva das features, usando somente anos anteriores
  3. Winsorização diagnóstica do target, usando somente labels já observáveis
  4. Z-score cross-section
  5. Score = -(w1*z_stock + w2*z_sales)  — sinal negativo = early-stage positivo

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

# Pesos calibrados pelo |IC| empírico sobre IEA 2026 (Cars, 2010-2025)
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
    ) -> pd.DataFrame:
        """
        Fixed-weight walk-forward evaluation. EVQTI weights remain pre-specified,
        while feature and target winsorization bounds are recalculated at each
        year using only earlier observations. This tests whether the fixed factor
        continues to hold out-of-sample without using future preprocessing bounds.

        For each window:
          - "train" years define the initial reference window; weights are not re-fit
          - "test" years are where IC is measured using point-in-time preprocessing

        This answers: "is the signal degrading over time?" rather than
        "can we always re-fit a signal that worked." The second question
        is trivially yes and tells you nothing about robustness.

        Parameters
        ----------
        train_years : size of the reference window (not used for fitting,
                      shown for context — EVQTI has no in-sample parameters
                      beyond the fixed, pre-calibrated weights)
        test_years  : number of years per test window
        step        : how many years to advance the window each iteration

        Returns
        -------
        pd.DataFrame
            Columns: window_start, window_end, ic_mean, ic_std, n_years, n_obs
        """
        panel = self._build_panel(df)
        years = sorted(panel["year"].unique())
        years = years[1:-1]  # drop first (no lag) and last (no target) year

        if len(years) < train_years + test_years:
            logger.warning(
                f"Not enough years for backtest: have {len(years)}, "
                f"need {train_years + test_years}."
            )
            return pd.DataFrame()

        rows = []
        start_idx = 0
        while start_idx + train_years + test_years <= len(years):
            test_window = years[start_idx + train_years : start_idx + train_years + test_years]

            window_ics = []
            window_n   = 0
            for yr in test_window:
                df_yr = self._compute_score(panel[panel["year"] == yr].copy())
                sub   = df_yr[["country", "evqti_score", "growth_t1_win"]].dropna()
                if len(sub) < 10:
                    continue
                ic_s, _ = stats.spearmanr(sub["evqti_score"], sub["growth_t1_win"])
                window_ics.append(ic_s)
                window_n += len(sub)

            if window_ics:
                rows.append({
                    "window_start": test_window[0],
                    "window_end":   test_window[-1],
                    "ic_mean":      round(float(np.mean(window_ics)), 3),
                    "ic_std":       round(float(np.std(window_ics)), 3) if len(window_ics) > 1 else 0.0,
                    "n_years":      len(window_ics),
                    "n_obs":        window_n,
                })

            start_idx += step

        return pd.DataFrame(rows)

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
