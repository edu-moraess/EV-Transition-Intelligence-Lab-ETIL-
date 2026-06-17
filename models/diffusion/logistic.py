# models/diffusion/logistic.py
"""
Modelo de Crescimento Logístico.
"""
import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class LogisticGrowth:
    """
    Modelo logístico: S(t) = M / (1 + exp(-k*(t - t0)))
    onde M é a capacidade de carga, k a taxa de crescimento, t0 o ponto de inflexão.
    """

    def __init__(self, M: Optional[float] = None, k: Optional[float] = None, t0: Optional[float] = None):
        self.M = M
        self.k = k
        self.t0 = t0
        self.fitted = False

    def cumulative(self, t: np.ndarray, M: float, k: float, t0: float) -> np.ndarray:
        return M / (1 + np.exp(-k * (t - t0)))

    def fit(self, t: np.ndarray, S: np.ndarray) -> Tuple[float, float, float]:
        """
        Estima M, k, t0.
        """
        M_guess = max(S) * 1.1
        k_guess = 0.5
        t0_guess = np.median(t)
        try:
            popt, _ = curve_fit(
                self.cumulative, t, S,
                p0=[M_guess, k_guess, t0_guess],
                bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=10000
            )
            self.M, self.k, self.t0 = popt
            self.fitted = True
            logger.info(f"Logistic: M={self.M:.2f}, k={self.k:.4f}, t0={self.t0:.2f}")
            return self.M, self.k, self.t0
        except Exception as e:
            logger.error(f"Erro no ajuste logístico: {e}")
            raise

    def predict(self, t: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Modelo não ajustado.")
        return self.cumulative(t, self.M, self.k, self.t0)

    def get_params(self) -> Dict[str, float]:
        return {"M": self.M, "k": self.k, "t0": self.t0}