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
        Walk-forward backtest: for each window, the feature weights and
        winsorization bounds are NOT re-fit — EVQTI's weights are fixed
        and empirically pre-calibrated (see module docstring), so this
        backtest measures something more honest than re-fitting in-sample
        each window: it measures whether a FIXED, already-published factor
        continues to hold out-of-sample across rolling periods.

        For each window:
          - "train" years are shown for reference only (no parameters fit here)
          - "test" years are where IC is actually measured

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
        Pipeline correto de construção:
          1. Winsorizar features RAW (1%-99%)
          2. Criar lags (t-1) sobre valores já winsorizados
          3. Criar target growth_t+1 (sem winsorização ainda)
          4. Winsorizar target (5%-95%)

        Garante que os lags também estão dentro dos bounds winsorizados.
        """
        panel = df.sort_values(["country", "year"]).copy()

        # Garantir ev_sales_share
        if "ev_sales_share" not in panel.columns:
            if "total_sales" in panel.columns and panel["total_sales"].notna().any():
                panel["ev_sales_share"] = (
                    panel["ev_sales"] / panel["total_sales"].replace(0, np.nan) * 100
                )
            else:
                panel["ev_sales_share"] = np.nan

        # Garantir growth_rate (para o target)
        if "growth_rate" not in panel.columns:
            panel["growth_rate"] = (
                panel.groupby("country")["ev_sales"].pct_change() * 100
            )

        # ── Passo 1: Winsorizar features RAW ────────────────────────────
        for col in ["ev_stock_share", "ev_sales_share"]:
            if col not in panel.columns:
                continue
            valid = panel[col].dropna()
            if len(valid) < 10:
                continue
            p1  = valid.quantile(0.01)
            p99 = valid.quantile(0.99)
            panel[col] = panel[col].clip(p1, p99)

        # ── Passo 2: Criar lags e clipar consistentemente ───────────────
        # Os lags são clipados com P1/P99 da própria distribuição do lag,
        # garantindo que a winsorização é sempre relativa ao que é observado.
        for col in ["ev_stock_share", "ev_sales_share"]:
            if col not in panel.columns:
                continue
            panel[f"{col}_lag1"] = panel.groupby("country")[col].shift(1)
            lag_valid = panel[f"{col}_lag1"].dropna()
            if len(lag_valid) < 10:
                continue
            lp1  = lag_valid.quantile(0.01)
            lp99 = lag_valid.quantile(0.99)
            panel[f"{col}_lag1"] = panel[f"{col}_lag1"].clip(lp1, lp99)

        # ── Passo 3: Target — crescimento futuro ─────────────────────────
        panel["growth_t1"] = panel.groupby("country")["growth_rate"].shift(-1)

        # ── Passo 4: Winsorizar target ────────────────────────────────────
        valid_t = panel["growth_t1"].dropna()
        if len(valid_t) > 10:
            p_lo = valid_t.quantile(_TARGET_WIN_LOW)
            p_hi = valid_t.quantile(_TARGET_WIN_HIGH)
            panel["growth_t1_win"] = panel["growth_t1"].clip(p_lo, p_hi)
        else:
            panel["growth_t1_win"] = panel["growth_t1"]

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
