# analytics/scenario_engine.py
"""
Motor de cenários para simular impactos de mudanças em parâmetros (ex: política, infraestrutura).
"""
import pandas as pd
import numpy as np
from models.diffusion import BassDiffusion
from models.forecasting import MonteCarloSimulator, BayesianGPForecast
from typing import Dict, List
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ScenarioEngine:
    """
    Aplica cenários alternativos alterando parâmetros de difusão.
    """

    def __init__(self, df: pd.DataFrame, country: str):
        self.df = df[df["country"] == country].sort_values("year").reset_index(drop=True)
        self.country = country

    def run_scenario(self, scenario_name: str, param_adjust: Dict[str, float]) -> pd.DataFrame:
        """
        Executa um cenário ajustando parâmetros do modelo Bass.

        Args:
            scenario_name (str): Nome descritivo.
            param_adjust (dict): Ajustes a aplicar nos parâmetros (ex: {"p": 1.5} para aumentar inovação).

        Returns:
            pd.DataFrame: Previsões para horizonte de 10 anos.
        """
        t_hist = self.df.index.values
        S_hist = self.df["ev_sales"].cumsum().values

        # Ajusta Bass
        bass = BassDiffusion()
        M, p, q = bass.fit(t_hist, S_hist)

        # Aplica ajustes
        if "p" in param_adjust:
            p = p * param_adjust["p"]
        if "q" in param_adjust:
            q = q * param_adjust["q"]
        if "M" in param_adjust:
            M = M * param_adjust["M"]

        # Prevê futuro
        horizon = 10
        t_future = np.arange(len(t_hist), len(t_hist) + horizon)
        S_future = bass.cumulative_adoption(t_future, M, p, q)

        # Converte para vendas anuais (diferença)
        sales_future = np.diff(S_future, prepend=S_hist[-1])

        # Cria DataFrame
        years = list(self.df["year"].values) + [self.df["year"].iloc[-1] + i + 1 for i in range(horizon)]
        df_scenario = pd.DataFrame({
            "year": years,
            "scenario": scenario_name,
            "ev_sales_cumulative": np.concatenate([S_hist, S_future]),
            "ev_sales_annual": np.concatenate([self.df["ev_sales"].values, sales_future])
        })
        return df_scenario