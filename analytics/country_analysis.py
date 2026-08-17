# analytics/country_analysis.py
"""
Análise detalhada de um país específico.
"""
from typing import Any

import numpy as np
import pandas as pd

from models.diffusion import BassDiffusion, GompertzModel, LogisticGrowth
from models.regimes import HMMRegimeDetector
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CountryAnalyzer:
    """
    Realiza análise completa para um país: difusão, regimes, métricas.
    """

    def __init__(self, df: pd.DataFrame, country: str):
        self.df = df[df["country"] == country].sort_values("year").reset_index(drop=True)
        self.country = country
        if self.df.empty:
            raise ValueError(f"País {country} não encontrado.")

    def get_data(self) -> pd.DataFrame:
        """Retorna dados históricos do país."""
        return self.df

    def fit_diffusion_models(self) -> dict[str, Any]:
        """
        Ajusta modelos de difusão e retorna parâmetros e previsões.
        """
        t = np.arange(1, len(self.df) + 1, dtype=float)
        S = self.df["ev_sales"].cumsum().values

        results = {}
        models = {
            "Bass": BassDiffusion(),
            "Logistic": LogisticGrowth(),
            "Gompertz": GompertzModel()
        }

        for name, model in models.items():
            try:
                model.fit(t, S)
                results[name] = {
                    "params": model.get_params(),
                    "fitted": True
                }
            except Exception as e:
                logger.warning(f"Falha no modelo {name} para {self.country}: {e}")
                results[name] = {"fitted": False, "error": str(e)}
        return results

    def detect_regimes(self) -> dict[str, Any]:
        """
        Detecta regimes usando o detector GMM de regimes.
        """
        if len(self.df) < 5:
            return {"error": "Dados insuficientes para o detector de regimes."}
        # Usa features: penetration e growth_rate
        features = []
        if "penetration" in self.df.columns and self.df["penetration"].notna().any():
            features.append(self.df["penetration"].fillna(0).values.reshape(-1, 1))
        if "growth_rate" in self.df.columns and self.df["growth_rate"].notna().any():
            features.append(self.df["growth_rate"].fillna(0).values.reshape(-1, 1))
        if not features:
            return {"error": "Features insuficientes."}
        X = np.hstack(features)
        regime_detector = HMMRegimeDetector(n_states=4)
        try:
            regime_detector.fit(X)
            states, probs = regime_detector.predict_regime(X)
            return {
                "states": states.tolist(),
                "probabilities": probs.tolist(),
                "transition_matrix": regime_detector.get_transition_matrix().tolist()
            }
        except Exception as e:
            logger.error(f"Detector de regimes falhou: {e}")
            return {"error": str(e)}