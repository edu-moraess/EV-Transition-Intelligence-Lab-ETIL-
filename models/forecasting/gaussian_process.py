"""
Previsão Bayesiana usando Gaussian Process Regression com tendência logística.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, RationalQuadratic, DotProduct
from typing import Tuple, Optional, Dict, Callable
from scipy.optimize import curve_fit
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BayesianGPForecast:
    """
    Regressão por Processo Gaussiano para previsão com incertezas.
    Suporta tendência logística (curva S) ou linear para capturar crescimento de longo prazo.
    """

    def __init__(self, 
                 kernel=None, 
                 alpha: float = 1e-6,
                 trend: str = "logistic",     # "logistic", "linear", ou None
                 trend_params: Optional[Dict] = None):
        """
        Args:
            kernel: kernel do GP (se None, usa RationalQuadratic + Linear)
            alpha: fator de regularização
            trend: tipo de tendência a ser subtraída ("logistic", "linear", None)
            trend_params: parâmetros iniciais para a curva (ex: {"L": 1000, "k": 0.3, "t0": 5})
        """
        if kernel is None:
            # Kernel mais adequado para séries com crescimento e ruído
            kernel = (ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) +
                      ConstantKernel(0.5) * DotProduct(sigma_0=1.0) +
                      WhiteKernel(0.1))
        self.kernel = kernel
        self.alpha = alpha
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, 
                                              n_restarts_optimizer=10, 
                                              normalize_y=True)
        self.fitted = False
        self.X_train = None
        self.y_train = None
        self.trend_type = trend
        self.trend_params = trend_params if trend_params else {}
        self._trend_func = None
        self._trend_popt = None

    def _logistic(self, t, L, k, t0):
        """Curva logística (S-curve)."""
        return L / (1 + np.exp(-k * (t - t0)))

    def _linear(self, t, a, b):
        """Tendência linear."""
        return a * t + b

    def _fit_trend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Ajusta a tendência escolhida aos dados."""
        t = X.flatten()
        if self.trend_type == "logistic":
            # Chutes iniciais razoáveis
            L_init = max(y) * 2.0 if max(y) > 0 else 1000
            k_init = 0.3
            t0_init = len(t) / 2
            try:
                popt, _ = curve_fit(self._logistic, t, y, 
                                    p0=[L_init, k_init, t0_init],
                                    maxfev=5000)
                self._trend_func = self._logistic
                self._trend_popt = popt
                logger.info(f"Tendência logística ajustada: L={popt[0]:.1f}, k={popt[1]:.3f}, t0={popt[2]:.1f}")
            except Exception as e:
                logger.warning(f"Falha no ajuste logístico, usando linear: {e}")
                self.trend_type = "linear"
                self._fit_trend(X, y)  # tenta linear como fallback
        elif self.trend_type == "linear":
            try:
                popt, _ = curve_fit(self._linear, t, y, p0=[1, 0])
                self._trend_func = self._linear
                self._trend_popt = popt
                logger.info(f"Tendência linear ajustada: a={popt[0]:.2f}, b={popt[1]:.1f}")
            except:
                logger.warning("Falha no ajuste linear, sem tendência.")
                self.trend_type = None
                self._trend_func = None
                self._trend_popt = None
        else:
            self._trend_func = None
            self._trend_popt = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta o GP aos dados, subtraindo a tendência (se especificada).

        Args:
            X (np.ndarray): Variáveis independentes (tempo) shape (n_samples, 1)
            y (np.ndarray): Variável alvo (adoção)
        """
        # Ajusta a tendência
        self._fit_trend(X, y)

        # Subtrai a tendência dos dados
        if self._trend_func is not None and self._trend_popt is not None:
            trend_vals = self._trend_func(X.flatten(), *self._trend_popt)
            y_detrended = y - trend_vals
        else:
            y_detrended = y

        # Treina o GP no resíduo
        self.model.fit(X, y_detrended)
        self.fitted = True
        self.X_train = X
        self.y_train = y
        logger.info("GP ajustado sobre o resíduo (tendência removida).")

    def predict(self, X_pred: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza previsão e retorna média e desvio padrão, somando a tendência de volta.

        Args:
            X_pred (np.ndarray): Pontos de previsão.
            return_std (bool): Se True, retorna também o desvio padrão.

        Returns:
            Tuple[np.ndarray, np.ndarray]: média, desvio padrão (se solicitado).
        """
        if not self.fitted:
            raise ValueError("Modelo não ajustado.")

        # Previsão do resíduo
        mean_res, std = self.model.predict(X_pred, return_std=True)

        # Soma a tendência de volta
        if self._trend_func is not None and self._trend_popt is not None:
            trend_future = self._trend_func(X_pred.flatten(), *self._trend_popt)
            mean = mean_res + trend_future
        else:
            mean = mean_res

        return mean, std
