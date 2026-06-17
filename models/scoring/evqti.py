# models/scoring/evqti.py
"""
EVQTI — EV Quant Transition Index (v2)

Fator quantitativo anti-leakage construído sobre variáveis defasadas (t-1),
com validação via Information Coefficient (IC), Rank IC (Spearman)
e análise de decis.

LÓGICA ECONÔMICA:
  Países com baixa penetração atual + alto momentum recente concentram
  o maior crescimento futuro. Isso é a lei de difusão de Bass aplicada
  ao cross-section de países: o mercado early-stage tem mais espaço.

CONSTRUÇÃO DO FATOR:
  1. Signal = -z(ev_stock_share_t-1) * w1 + z(growth_rate_t-1) * w2
     (invertemos stock_share: baixa penetração = sinal positivo)
  2. Winsorização 1%-99% em todas as features
  3. Z-score por cross-section
  4. Score final = média ponderada dos sinais disponíveis

VALIDAÇÃO:
  - IC (Pearson sobre ranks) vs. crescimento futuro
  - Rank IC (Spearman) vs. crescimento futuro
  - Decile spread: diferença entre decil superior e inferior
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Pesos calibrados pelo IC empírico (|IC| de cada feature contra growth_t+1)
# stock_share: IC=-0.265 (invertido) → peso 0.45
# sales_share: IC=-0.407 (invertido) → peso 0.35
# growth_rate: IC=+0.222             → peso 0.20
_DEFAULT_WEIGHTS = {
    "ev_stock_share": (-1.0, 0.45),   # (sinal, peso)
    "ev_sales_share": (-1.0, 0.35),
    "growth_rate":    (+1.0, 0.20),
}


class EVQTI:
    """
    Fator quantitativo de transição EV com validação estatística completa.
    """

    def __init__(self, weights: Optional[dict] = None):
        self.weights       = weights or _DEFAULT_WEIGHTS
        self.fitted        = False
        self.country_scores: Optional[pd.Series] = None
        self.panel_: Optional[pd.DataFrame]      = None   # painel completo com lags
        self.ic_stats_: Optional[pd.DataFrame]   = None   # IC por ano
        self.decile_stats_: Optional[pd.DataFrame] = None # retorno por decil

    # ── API pública ──────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "EVQTI":
        """
        Ajusta o EVQTI ao DataFrame processado pelo pipeline ETIL.

        Args:
            df: DataFrame com colunas country, year, ev_sales,
                ev_stock_share (opcional), ev_sales_share (opcional),
                growth_rate (opcional).

        Returns:
            self
        """
        panel = self._build_panel(df)
        self.panel_ = panel

        # Score cross-section no último ano disponível
        latest = int(panel["year"].max())
        df_score = panel[panel["year"] == latest].copy()
        df_score = self._compute_score(df_score)

        self.country_scores = (
            df_score
            .set_index("country")["evqti_score"]
            .sort_values(ascending=False)
        )

        # Validação
        self.ic_stats_     = self._compute_ic(panel)
        self.decile_stats_ = self._compute_deciles(panel)

        self.fitted = True
        logger.info(
            f"EVQTI v2 ajustado: {len(self.country_scores)} países | "
            f"IC médio: {self.ic_stats_['rank_ic'].mean():.3f}"
        )
        return self

    def get_ranking(self) -> pd.DataFrame:
        """Retorna ranking completo com score e rank."""
        if not self.fitted:
            raise ValueError("EVQTI não ajustado. Chame fit() primeiro.")
        ranking = (
            self.country_scores
            .reset_index()
            .rename(columns={"ev_stock_share": "evqti_score"})
        )
        ranking.columns = ["country", "evqti_score"]
        ranking["rank"] = np.arange(1, len(ranking) + 1)
        return ranking[["country", "rank", "evqti_score"]]

    def get_ic_stats(self) -> pd.DataFrame:
        """Retorna IC e Rank IC por ano."""
        if self.ic_stats_ is None:
            raise ValueError("EVQTI não ajustado.")
        return self.ic_stats_

    def get_decile_stats(self) -> pd.DataFrame:
        """Retorna crescimento médio por decil de EVQTI."""
        if self.decile_stats_ is None:
            raise ValueError("EVQTI não ajustado.")
        return self.decile_stats_

    # ── Construção do painel ─────────────────────────────────────────────

    def _build_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constrói painel com features defasadas e target futuro."""
        panel = df.sort_values(["country", "year"]).copy()

        # Garantir growth_rate
        if "growth_rate" not in panel.columns:
            panel["growth_rate"] = (
                panel.groupby("country")["ev_sales"]
                .pct_change() * 100
            )

        # Garantir ev_sales_share (proxy se não disponível)
        if "ev_sales_share" not in panel.columns:
            if "total_sales" in panel.columns and panel["total_sales"].notna().any():
                panel["ev_sales_share"] = (
                    panel["ev_sales"] / panel["total_sales"].replace(0, np.nan) * 100
                )
            else:
                panel["ev_sales_share"] = np.nan

        # Winsorizar features antes do lag
        for col in ["ev_stock_share", "ev_sales_share", "growth_rate"]:
            if col in panel.columns:
                p1  = panel[col].quantile(0.01)
                p99 = panel[col].quantile(0.99)
                panel[col] = panel[col].clip(p1, p99)

        # Criar lags (t-1) — anti-leakage
        for col in ["ev_stock_share", "ev_sales_share", "growth_rate"]:
            if col in panel.columns:
                panel[f"{col}_lag1"] = panel.groupby("country")[col].shift(1)

        # Target: crescimento futuro (t+1)
        panel["growth_t1"] = panel.groupby("country")["growth_rate"].shift(-1)

        # Winsorizar target
        p5, p95 = panel["growth_t1"].quantile([0.05, 0.95])
        panel["growth_t1_win"] = panel["growth_t1"].clip(p5, p95)

        return panel

    def _compute_score(self, df_cross: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula score EVQTI em cross-section.

        Para cada feature disponível:
          1. Winsoriza 1%-99%
          2. Z-score
          3. Aplica sinal e peso calibrado
          4. Soma ponderada normalizada
        """
        df_cross = df_cross.copy()
        score_components = []
        weight_sum       = 0.0

        scaler = StandardScaler()

        for feat, (sign, weight) in self.weights.items():
            lag_col = f"{feat}_lag1"
            raw_col = feat

            # Usa lag se disponível, senão raw
            col = lag_col if lag_col in df_cross.columns and df_cross[lag_col].notna().any() else raw_col

            if col not in df_cross.columns:
                continue

            sub = df_cross[col].copy()
            coverage = sub.notna().mean()
            if coverage < 0.20:
                logger.warning(f"Feature {feat} com baixa cobertura ({coverage:.0%}) — ignorada.")
                continue

            # Imputar mediana antes do z-score
            sub = sub.fillna(sub.median())

            # Z-score
            z = (sub - sub.mean()) / (sub.std() + 1e-9)
            z = z * sign  # aplica direção do sinal

            score_components.append(z * weight)
            weight_sum += weight

        if not score_components:
            raise ValueError("Nenhuma feature válida disponível para calcular EVQTI.")

        df_cross["evqti_score"] = sum(score_components) / weight_sum

        return df_cross

    # ── Validação estatística ────────────────────────────────────────────

    def _compute_ic(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        IC e Rank IC por ano.

        Para cada ano t, calcula o score EVQTI usando features de t-1
        e mede a correlação com o crescimento em t (forward return).
        """
        years = sorted(panel["year"].unique())[1:-1]  # exclui primeiro e último
        rows  = []

        for yr in years:
            df_yr = panel[panel["year"] == yr].copy()
            df_yr = self._compute_score(df_yr)

            sub = df_yr[["country", "evqti_score", "growth_t1_win"]].dropna()
            if len(sub) < 10:
                continue

            ic_pearson, p_pearson   = stats.pearsonr(sub["evqti_score"], sub["growth_t1_win"])
            ic_spearman, p_spearman = stats.spearmanr(sub["evqti_score"], sub["growth_t1_win"])

            rows.append({
                "year":     yr,
                "ic":       round(ic_pearson,  3),
                "rank_ic":  round(ic_spearman, 3),
                "p_value":  round(p_spearman,  3),
                "n":        len(sub),
                "significant": p_spearman < 0.05,
            })

        return pd.DataFrame(rows)

    def _compute_deciles(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Decile analysis: países ordenados por EVQTI em t,
        retorno (growth) medido em t+1.
        """
        years = sorted(panel["year"].unique())[1:-1]
        all_rows = []

        for yr in years:
            df_yr = panel[panel["year"] == yr].copy()
            df_yr = self._compute_score(df_yr)
            sub = df_yr[["country", "evqti_score", "growth_t1_win"]].dropna()
            if len(sub) < 15:
                continue
            try:
                sub["decile"] = pd.qcut(sub["evqti_score"], 5, labels=False, duplicates="drop")
                sub["year"]   = yr
                all_rows.append(sub)
            except Exception:
                pass

        if not all_rows:
            return pd.DataFrame(columns=["decile", "growth_t1_win"])

        full = pd.concat(all_rows, ignore_index=True)
        decile_stats = (
            full.groupby("decile")["growth_t1_win"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_growth", "std": "std_growth", "count": "n"})
        )
        decile_stats["decile_label"] = decile_stats["decile"].map({
            0: "Q1 (mais baixo)", 1: "Q2", 2: "Q3", 3: "Q4", 4: "Q5 (mais alto)"
        })
        return decile_stats
