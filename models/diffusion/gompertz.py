# models/diffusion/gompertz.py
"""
Modelo de Gompertz: S(t) = M * exp(-exp(-k*(t - t0)))
"""
import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class GompertzModel:
    """
    Modelo de Gompertz para crescimento assimétrico.
    """

    def __init__(self, M: Optional[float] = None, k: Optional[float] = None, t0: Optional[float] = None):
        self.M = M
        self.k = k
        self.t0 = t0
        self.fitted = False

    def cumulative(self, t: np.ndarray, M: float, k: float, t0: float) -> np.ndarray:
        return M * np.exp(-np.exp(-k * (t - t0)))

    def fit(self, t: np.ndarray, S: np.ndarray) -> Tuple[float, float, float]:
        M_guess = max(S) * 1.1
        k_guess = 0.3
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
            logger.info(f"Gompertz: M={self.M:.2f}, k={self.k:.4f}, t0={self.t0:.2f}")
            return self.M, self.k, self.t0
        except Exception as e:
            logger.error(f"Erro no ajuste Gompertz: {e}")
            raise

    def predict(self, t: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Modelo não ajustado.")
        return self.cumulative(t, self.M, self.k, self.t0)

    def get_params(self) -> Dict[str, float]:
        return {"M": self.M, "k": self.k, "t0": self.t0}