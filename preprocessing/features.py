# preprocessing/features.py
"""
Engenharia de features quantitativas.
"""
import pandas as pd
import numpy as np
from utils.helpers import growth_rate, cagr, rolling_momentum
from utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    """
    Cria features derivadas para análise de transição.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def engineer(self) -> pd.DataFrame:
        """
        Executa todas as etapas de criação de features.

        Returns:
            pd.DataFrame: DataFrame com novas colunas.
        """
        self._add_penetration()
        self._add_growth_rates()
        self._add_cagr()
        self._add_momentum()
        self._add_volatility()
        self._add_infrastructure_pressure()
        self._add_ev_per_capita()
        return self.df

    def _add_penetration(self) -> None:
        """Calcula a penetração de EV: ev_sales / total_sales * 100."""
        if "total_sales" in self.df.columns and self.df["total_sales"].notna().any():
            self.df["penetration"] = (self.df["ev_sales"] / self.df["total_sales"]) * 100
            self.df["penetration"].fillna(0, inplace=True)
        else:
            logger.warning("total_sales não disponível, penetração não calculada.")

    def _add_growth_rates(self) -> None:
        """Adiciona taxa de crescimento anual de ev_sales."""
        self.df["growth_rate"] = self.df.groupby("country")["ev_sales"].transform(growth_rate)

    def _add_cagr(self) -> None:
        """Adiciona CAGR para cada país (constante ao longo dos anos)."""
        cagr_vals = self.df.groupby("country")["ev_sales"].agg(cagr)
        self.df["cagr"] = self.df["country"].map(cagr_vals)

    def _add_momentum(self) -> None:
        """Adiciona momentum (3 anos)."""
        self.df["momentum_3y"] = self.df.groupby("country")["ev_sales"].transform(
            lambda x: rolling_momentum(x, window=3)
        )

    def _add_volatility(self) -> None:
        """Adiciona volatilidade (desvio padrão do crescimento)."""
        self.df["volatility"] = self.df.groupby("country")["growth_rate"].transform(
            lambda x: x.rolling(4, min_periods=2).std()
        )

    def _add_infrastructure_pressure(self) -> None:
        """Calcula EV por ponto de carga (se disponível)."""
        if "charging_points" in self.df.columns and self.df["charging_points"].notna().any():
            self.df["ev_per_charger"] = self.df["ev_sales"] / (self.df["charging_points"] + 1)
        else:
            self.df["ev_per_charger"] = np.nan

    def _add_ev_per_capita(self) -> None:
        """Calcula EV per capita (se população disponível)."""
        if "population" in self.df.columns and self.df["population"].notna().any():
            self.df["ev_per_capita"] = self.df["ev_sales"] / (self.df["population"] + 1)
        else:
            self.df["ev_per_capita"] = np.nan