"""
EV Quant Transition Index (EVQTI)

Índice proprietário combinando:
- Seleção robusta de features
- Imputação de valores ausentes
- Padronização Z-score
- PCA para extração do fator principal de transição EV
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
    Calcula o EV Quant Transition Index para cada país
    com base em múltiplas métricas da transição para
    veículos elétricos.
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
        self.used_features = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Ajusta o EVQTI usando os dados históricos.
        """

        # Seleciona apenas métricas que possuem dados suficientes
        available_features = [
            feature for feature in self.features
            if (
                feature in df.columns
                and df[feature].notna().mean() >= 0.10
            )
        ]

        if len(available_features) == 0:
            raise ValueError(
                "Nenhuma feature válida disponível para calcular o EVQTI."
            )

        logger.info(
            f"Features candidatas EVQTI: {available_features}"
        )

        # Média histórica por país
        X = (
            df
            .groupby("country")[available_features]
            .mean()
        )

        logger.info(
            f"Países antes da limpeza: {len(X)}"
        )

        # Remove métricas com pouca cobertura entre países
        coverage = X.notna().mean()

        valid_features = coverage[
            coverage >= 0.30
        ].index.tolist()

        X = X[valid_features]

        if X.shape[1] == 0:
            raise ValueError(
                "Nenhuma métrica possui cobertura suficiente após agregação."
            )

        self.used_features = valid_features

        logger.info(
            f"Features finais EVQTI: {valid_features}"
        )

        # Imputação robusta pela mediana
        missing_before = X.isna().sum().sum()

        X = X.fillna(
            X.median(numeric_only=True)
        )

        logger.info(
            f"Valores ausentes imputados: {missing_before}"
        )

        # Padronização Z-score
        X_scaled = self.scaler.fit_transform(X)

        logger.info(
            f"Calculando PCA com {len(X)} países "
            f"e {X.shape[1]} métricas."
        )

        # PCA
        self.pca_score.fit(X_scaled)

        scores = self.pca_score.transform(X_scaled)

        # Cria ranking
        self.country_scores = (
            pd.Series(
                scores,
                index=X.index,
                name="evqti_score"
            )
            .sort_values(ascending=False)
        )

        self.fitted = True

        logger.info(
            f"EVQTI concluído com {len(self.country_scores)} países."
        )

    def get_ranking(self) -> pd.DataFrame:
        """
        Retorna ranking final dos países.
        """

        if not self.fitted:
            raise ValueError(
                "EVQTI ainda não foi ajustado."
            )

        ranking = (
            self.country_scores
            .reset_index()
            .rename(
                columns={
                    "index": "country",
                    "evqti_score": "evqti_score"
                }
            )
        )

        ranking["rank"] = (
            np.arange(len(ranking)) + 1
        )

        return ranking[
            [
                "country",
                "rank",
                "evqti_score"
            ]
        ]