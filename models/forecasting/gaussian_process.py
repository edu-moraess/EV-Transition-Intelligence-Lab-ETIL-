# models/forecasting/gaussian_process.py
"""
Previsão Bayesiana usando Gaussian Process Regression.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from typing import Tuple, Optional, Dict
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BayesianGPForecast:
    """
    Regressão por Processo Gaussiano para previsão com incertezas.
    """

    def __init__(self, kernel=None, alpha: float = 1e-6):
        if kernel is None:
            kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(1.0, (0.1, 10.0)) + WhiteKernel(1.0)
        self.kernel = kernel
        self.alpha = alpha
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10, normalize_y=True)
        self.fitted = False
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta o GP.

        Args:
            X (np.ndarray): Variáveis independentes (tempo, etc.) shape (n_samples, n_features).
            y (np.ndarray): Variável alvo (adoção).
        """
        self.model.fit(X, y)
        self.fitted = True
        self.X_train = X
        self.y_train = y
        logger.info("GP ajustado.")

    def predict(self, X_pred: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza previsão e retorna média e desvio padrão.

        Args:
            X_pred (np.ndarray): Pontos de previsão.
            return_std (bool): Se True, retorna também o desvio padrão.

        Returns:
            Tuple[np.ndarray, np.ndarray]: média, desvio padrão (se solicitado).
        """
        if not self.fitted:
            raise ValueError("Modelo não ajustado.")
        mean, std = self.model.predict(X_pred, return_std=True)
        return mean, std