# models/forecasting/monte_carlo.py
"""
Simulação de Monte Carlo para cenários futuros.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from scipy.stats import norm
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MonteCarloSimulator:
    """
    Gera múltiplas trajetórias futuras a partir de um modelo de previsão (GP) e parâmetros de incerteza.
    """

    def __init__(self, n_simulations: int = 5000, horizon: int = 10, confidence_level: float = 0.95):
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.confidence_level = confidence_level
        self.simulations = None

    def simulate_from_gp(self, gp_model, X_future: np.ndarray) -> np.ndarray:
        """
        Gera simulações a partir de um modelo GP.

        Args:
            gp_model: instância de BayesianGPForecast já ajustada.
            X_future (np.ndarray): Pontos futuros.

        Returns:
            np.ndarray: Array de shape (n_simulations, len(X_future)) com trajetórias.
        """
        mean, std = gp_model.predict(X_future, return_std=True)
        # Gera amostras normais multivariadas: assumimos independência entre pontos (simplificação)
        sims = np.random.normal(loc=mean, scale=std, size=(self.n_simulations, len(X_future)))
        # Ajusta para crescimento positivo? Pode-se aplicar transformações.
        self.simulations = sims
        logger.info(f"Simulação MC: {self.n_simulations} trajetórias, horizonte {len(X_future)}.")
        return sims

    def get_percentiles(self, sims: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula percentis (lower, median, upper) para cada ponto do horizonte.

        Args:
            sims (np.ndarray): Array de simulações (n_sim, horizon).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: lower, median, upper.
        """
        lower = np.percentile(sims, (1 - self.confidence_level) / 2 * 100, axis=0)
        median = np.percentile(sims, 50, axis=0)
        upper = np.percentile(sims, (1 + self.confidence_level) / 2 * 100, axis=0)
        return lower, median, upper

    def scenario_adjust(self, sims: np.ndarray, scenario: str = "base") -> np.ndarray:
        """
        Ajusta as simulações para cenários otimista/pessimista (multiplicador).
        Cenários: 'base' (1.0), 'optimistic' (1.2), 'pessimistic' (0.8).
        """
        factor = {"base": 1.0, "optimistic": 1.2, "pessimistic": 0.8}.get(scenario, 1.0)
        return sims * factor