# models/scoring/evqti.py
"""
EV Quant Transition Index (EVQTI): índice proprietário combinando PCA, z-score e ponderação por entropia.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .pca_score import PCAScore
from typing import List, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class EVQTI:
    """
    Calcula o EVQTI para cada país com base em métricas quantitativas.
    """

    def __init__(self, features: List[str] = None):
        self.features = features or [
            "penetration",
            "growth_rate",
            "cagr",
            "momentum_3y",
            "ev_per_capita",
            "ev_per_charger"
        ]
        self.scaler = StandardScaler()
        self.pca_score = PCAScore()
        self.fitted = False
        self.country_scores = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Ajusta o índice a partir dos dados.
        """
        # Seleciona features
        available = [f for f in self.features if f in df.columns and df[f].notna().any()]
        if not available:
            raise ValueError("Nenhuma feature disponível para EVQTI.")
        logger.info(f"Features usadas: {available}")

        X = df.groupby("country")[available].mean()  # média por país para estabilidade
        # Padroniza
        X_scaled = self.scaler.fit_transform(X)
        # PCA
        self.pca_score.fit(X)
        self.fitted = True
        # Calcula score para cada país
        scores = self.pca_score.transform(X)
        self.country_scores = pd.Series(scores, index=X.index).sort_values(ascending=False)
        logger.info("EVQTI calculado.")

    def get_ranking(self) -> pd.DataFrame:
        """
        Retorna ranking dos países com seus scores EVQTI.
        """
        if not self.fitted:
            raise ValueError("EVQTI não ajustado.")
        df_ranking = pd.DataFrame({
            "country": self.country_scores.index,
            "evqti_score": self.country_scores.values
        }).sort_values("evqti_score", ascending=False)
        df_ranking["rank"] = range(1, len(df_ranking) + 1)
        return df_ranking 