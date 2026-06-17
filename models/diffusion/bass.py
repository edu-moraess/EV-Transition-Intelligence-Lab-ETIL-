# models/diffusion/bass.py
"""
Modelo de Difusão de Bass: estima parâmetros p (inovação), q (imitação), M (potencial).
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from typing import Tuple, Optional, Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BassDiffusion:
    """
    Modelo de Bass para prever adoção cumulativa de tecnologia.

    A forma contínua: S(t) = M * (1 - exp(-(p+q)*t)) / (1 + (q/p)*exp(-(p+q)*t))
    onde S(t) é a adoção cumulativa no tempo t, M é o mercado potencial,
    p é o coeficiente de inovação, q é o coeficiente de imitação.
    """

    def __init__(self, M: Optional[float] = None, p: Optional[float] = None, q: Optional[float] = None):
        self.M = M
        self.p = p
        self.q = q
        self.fitted = False

    def cumulative_adoption(self, t: np.ndarray, M: float, p: float, q: float) -> np.ndarray:
        """Função de adoção cumulativa de Bass."""
        exponent = -(p + q) * t
        numerator = M * (1 - np.exp(exponent))
        denominator = 1 + (q / p) * np.exp(exponent) if p != 0 else 1
        return numerator / denominator

    def fit(self, t: np.ndarray, S: np.ndarray) -> Tuple[float, float, float]:
        """
        Estima parâmetros M, p, q a partir de dados de adoção cumulativa.

        Args:
            t (np.ndarray): Tempo (anos, indexado a partir de 0).
            S (np.ndarray): Adoção cumulativa observada.

        Returns:
            Tuple[float, float, float]: M, p, q estimados.
        """
        # Chute inicial razoável
        M_guess = max(S) * 1.2 if max(S) > 0 else 1.0
        p_guess = 0.01
        q_guess = 0.3

        def residuals(params, t, y):
            M, p, q = params
            try:
                y_pred = self.cumulative_adoption(t, M, p, q)
                return (y_pred - y) ** 2
            except:
                return np.ones_like(y) * 1e6

        # Primeiro, tentamos curve_fit
        try:
            popt, _ = curve_fit(
                lambda t, M, p, q: self.cumulative_adoption(t, M, p, q),
                t, S,
                p0=[M_guess, p_guess, q_guess],
                bounds=([0, 0, 0], [np.inf, 1, 1]),
                maxfev=10000
            )
            self.M, self.p, self.q = popt
        except Exception as e:
            logger.warning(f"Curve_fit falhou: {e}. Usando minimização.")
            # Minimização por Nelder-Mead
            result = minimize(
                lambda params: np.sum(residuals(params, t, S)),
                [M_guess, p_guess, q_guess],
                bounds=[(0, None), (0, 1), (0, 1)],
                method='L-BFGS-B'
            )
            if result.success:
                self.M, self.p, self.q = result.x
            else:
                raise RuntimeError("Falha na estimação dos parâmetros de Bass.")

        self.fitted = True
        logger.info(f"Bass: M={self.M:.2f}, p={self.p:.4f}, q={self.q:.4f}")
        return self.M, self.p, self.q

    def predict(self, t: np.ndarray) -> np.ndarray:
        """
        Gera previsão de adoção cumulativa para os tempos fornecidos.
        """
        if not self.fitted:
            raise ValueError("Modelo não ajustado. Execute fit() primeiro.")
        return self.cumulative_adoption(t, self.M, self.p, self.q)

    def get_params(self) -> Dict[str, float]:
        """Retorna dicionário com parâmetros."""
        return {"M": self.M, "p": self.p, "q": self.q}