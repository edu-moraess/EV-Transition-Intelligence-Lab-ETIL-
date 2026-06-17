"""
EV Quant Transition Index (EVQTI)

Índice quantitativo baseado em:
- Seleção de métricas com cobertura adequada
- Agregação histórica por país
- Imputação robusta por mediana
- PCA para extração do fator principal de transição EV
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from .pca_score import PCAScore
from utils.logger import setup_logger


logger = setup_logger(__name__)


class EVQTI:
    """
    Calcula o EV Quant Transition Index para países.
    """

    def __init__(self, features: Optional[List[str]] = None):

        self.features = features or [
            "penetration",
            "growth_rate",
            "cagr",
            "momentum_3y",
            "ev_per_capita",
            "ev_per_charger",
        ]

        self.pca_score = PCAScore()

        self.fitted = False
        self.country_scores = None
        self.used_features = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Ajusta o EVQTI usando dados históricos.
        """

        # Seleciona apenas métricas que possuem dados mínimos
        available_features = [
            feature
            for feature in self.features
            if (
                feature in df.columns
                and df[feature].notna().mean() >= 0.10
            )
        ]

        if not available_features:
            raise ValueError(
                "Nenhuma métrica válida disponível para o EVQTI."
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
            f"Países encontrados: {len(X)}"
        )

        # Remove métricas com baixa cobertura após agregação
        coverage = X.notna().mean()

        valid_features = (
            coverage[coverage >= 0.30]
            .index
            .tolist()
        )

        X = X[valid_features]

        if X.empty:
            raise ValueError(
                "Nenhuma métrica possui cobertura suficiente."
            )

        self.used_features = valid_features

        logger.info(
            f"Features finais do EVQTI: {valid_features}"
        )

        # Imputação por mediana para preservar países
        missing_values = X.isna().sum().sum()

        X = X.fillna(
            X.median(numeric_only=True)
        )

        logger.info(
            f"Valores ausentes preenchidos: {missing_values}"
        )

        logger.info(
            f"Executando PCA com {len(X)} países e "
            f"{X.shape[1]} variáveis."
        )

        # PCA + Z-score é feito dentro do PCAScore
        self.pca_score.fit(X)

        scores = self.pca_score.transform(X)

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
        Retorna o ranking final do EVQTI.
        """

        if not self.fitted:
            raise ValueError(
                "EVQTI ainda não foi ajustado."
            )

        ranking = (
            self.country_scores
            .reset_index()
        )

        ranking.columns = [
            "country",
            "evqti_score",
        ]

        ranking["rank"] = np.arange(
            1,
            len(ranking) + 1
        )

        return ranking[
            [
                "country",
                "rank",
                "evqti_score",
            ]
        ]