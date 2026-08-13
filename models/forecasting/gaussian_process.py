"""Gaussian-process forecasting with optional deterministic trend removal."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, RationalQuadratic, WhiteKernel

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BayesianGPForecast:
    """Fit a Gaussian process to a univariate time series and sample correlated paths."""

    def __init__(
        self,
        kernel=None,
        alpha: float = 1e-6,
        trend: str | None = None,
        trend_params: dict | None = None,
        random_state: int | None = 42,
    ):
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if trend not in {None, "linear", "logistic"}:
            raise ValueError("trend must be None, 'linear' or 'logistic'.")
        if kernel is None:
            kernel = (
                ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
                + ConstantKernel(0.5) * DotProduct(sigma_0=1.0)
                + WhiteKernel(0.1)
            )
        self.kernel = kernel
        self.alpha = alpha
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=random_state,
        )
        self.fitted = False
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.trend_type = trend
        self.trend_params = trend_params or {}
        self._trend_func: Callable | None = None
        self._trend_popt: np.ndarray | None = None

    @staticmethod
    def _logistic(t: np.ndarray, L: float, k: float, t0: float) -> np.ndarray:
        return L / (1 + np.exp(-np.clip(k * (t - t0), -700, 700)))

    @staticmethod
    def _linear(t: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * t + b

    def _fit_trend(self, X: np.ndarray, y: np.ndarray) -> None:
        self._trend_func = None
        self._trend_popt = None
        if self.trend_type is None:
            return
        t = X.reshape(-1)
        try:
            if self.trend_type == "linear":
                parameters, _ = curve_fit(self._linear, t, y, p0=[0.0, float(np.mean(y))])
                self._trend_func = self._linear
                self._trend_popt = parameters
                return
            lower = [max(float(np.max(y)), 1e-8), 1e-8, float(np.min(t) - 2 * np.ptp(t) - 1)]
            upper = [max(float(np.max(y)) * 100, 1.0), 5.0, float(np.max(t) + 2 * np.ptp(t) + 1)]
            initial = [max(float(np.max(y)) * 1.5, lower[0] * 1.01), 0.3, float(np.median(t))]
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                parameters, _ = curve_fit(
                    self._logistic,
                    t,
                    y,
                    p0=initial,
                    bounds=(lower, upper),
                    maxfev=20_000,
                )
            self._trend_func = self._logistic
            self._trend_popt = parameters
        except (RuntimeError, ValueError, OptimizeWarning) as error:
            logger.warning("Trend fit failed; forecasting residual-free GP instead: %s", error)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianGPForecast":
        """Fit the process on finite, aligned observations."""
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        if X_array.ndim != 2 or len(X_array) != len(y_array) or len(y_array) < 4:
            raise ValueError("GP fitting requires at least four aligned two-dimensional inputs.")
        if not np.all(np.isfinite(X_array)) or not np.all(np.isfinite(y_array)):
            raise ValueError("GP inputs must be finite.")
        self._fit_trend(X_array, y_array)
        trend = self._trend_values(X_array)
        self.model.fit(X_array, y_array - trend)
        self.fitted = True
        self.X_train = X_array
        self.y_train = y_array
        return self

    def predict(self, X_pred: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean and marginal standard deviation at future inputs."""
        self._check_fitted()
        inputs = self._validate_prediction_input(X_pred)
        mean_residual, std = self.model.predict(inputs, return_std=True)
        return mean_residual + self._trend_values(inputs), std

    def sample_paths(self, X_pred: np.ndarray, n_samples: int, seed: int | None = None) -> np.ndarray:
        """Draw correlated posterior paths using the Gaussian-process covariance matrix."""
        self._check_fitted()
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        inputs = self._validate_prediction_input(X_pred)
        mean_residual, covariance = self.model.predict(inputs, return_cov=True)
        covariance = np.asarray(covariance, dtype=float)
        jitter = max(float(np.max(np.diag(covariance))) * 1e-10, 1e-12)
        covariance = covariance + np.eye(len(inputs)) * jitter
        rng = np.random.default_rng(seed)
        paths = rng.multivariate_normal(mean_residual, covariance, size=n_samples)
        return paths + self._trend_values(inputs)

    def _trend_values(self, X: np.ndarray) -> np.ndarray:
        if self._trend_func is None or self._trend_popt is None:
            return np.zeros(len(X), dtype=float)
        return self._trend_func(X.reshape(-1), *self._trend_popt)

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

    @staticmethod
    def _validate_prediction_input(X_pred: np.ndarray) -> np.ndarray:
        inputs = np.asarray(X_pred, dtype=float)
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        if inputs.ndim != 2 or len(inputs) == 0 or not np.all(np.isfinite(inputs)):
            raise ValueError("Prediction inputs must be a non-empty finite two-dimensional array.")
        return inputs
