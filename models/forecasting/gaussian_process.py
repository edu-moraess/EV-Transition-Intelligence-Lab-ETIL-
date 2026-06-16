# models/forecasting/gaussian_process.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from utils.logger import setup_logger

logger = setup_logger(__name__)

class GaussianProcessForecast:
    """
    Previsão usando Gaussian Process.
    """

    def __init__(self, n_samples: int = 100):
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
        self.fitted = False

    def fit(self, t: np.ndarray, y: np.ndarray) -> None:
        t = t.reshape(-1, 1)
        self.gp.fit(t, y)
        self.fitted = True
        logger.info("Gaussian Process ajustado.")

    def predict(self, t_future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Modelo não ajustado.")
        t_future = t_future.reshape(-1, 1)
        y_pred, y_std = self.gp.predict(t_future, return_std=True)
        return y_pred, y_std
