# models/regimes/changepoint.py
"""
Detecção de mudanças de regime via Bayesian Change Point.
Simplificação: usa a biblioteca `ruptures`? Mas manteremos puramente scipy/numpy.
Implementaremos um método simplificado baseado em soma cumulativa (CUSUM) com abordagem Bayesiana.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BayesianChangePoint:
    """
    Detecta pontos de mudança em uma série temporal usando uma abordagem Bayesiana simplificada.
    Aqui, calculamos a probabilidade de mudança em cada ponto usando uma métrica de verossimilhança.
    """

    def __init__(self, min_distance: int = 2):
        self.min_distance = min_distance
        self.changepoints = []

    def detect(self, series: np.ndarray) -> List[int]:
        """
        Detecta pontos de mudança baseados em mudanças de média/variancia.

        Args:
            series (np.ndarray): Série temporal unidimensional.

        Returns:
            List[int]: Índices dos pontos de mudança.
        """
        n = len(series)
        if n < 3:
            return []

        # Usa uma abordagem de janela: calcula a diferença entre médias antes/depois
        scores = np.zeros(n)
        for i in range(self.min_distance, n - self.min_distance):
            left = series[:i]
            right = series[i:]
            mean_left = np.mean(left)
            mean_right = np.mean(right)
            var_left = np.var(left)
            var_right = np.var(right)
            # Score baseado na diferença de médias normalizada (efetivamente t-stat)
            if var_left + var_right > 0:
                score = (mean_left - mean_right) ** 2 / (var_left + var_right)
            else:
                score = 0
            scores[i] = score

        # Pega os picos acima de um limiar (adaptativo)
        threshold = np.percentile(scores, 95) if np.any(scores) else 0
        changepoints = []
        for i in range(self.min_distance, n - self.min_distance):
            if scores[i] > threshold and scores[i] == max(scores[max(0,i-2):min(n,i+3)]):
                changepoints.append(i)

        self.changepoints = changepoints
        logger.info(f"Changepoints detectados: {changepoints}")
        return changepoints

    def get_probabilities(self, series: np.ndarray) -> np.ndarray:
        """
        Retorna uma probabilidade aproximada de mudança em cada ponto (normalizada).
        """
        n = len(series)
        probs = np.zeros(n)
        if n < 3:
            return probs
        for i in range(self.min_distance, n - self.min_distance):
            left = series[:i]
            right = series[i:]
            mean_left = np.mean(left)
            mean_right = np.mean(right)
            var_left = np.var(left) + 1e-6
            var_right = np.var(right) + 1e-6
            # Log-verossimilhança aproximada
            ll = -0.5 * (np.log(var_left) + np.log(var_right) + (mean_left - mean_right)**2/(var_left+var_right))
            probs[i] = np.exp(ll)  # não normalizado
        # Normaliza
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        return probs