# models/scoring/pca_score.py
"""
Cálculo de score via PCA para composição de indicadores.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class PCAScore:
    """
    Aplica PCA para extrair um score composto a partir de múltiplas features.
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        self.fitted = False
        self.explained_variance = None

    def fit(self, X: pd.DataFrame) -> None:
        """
        Ajusta PCA aos dados.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=self.n_components or min(X.shape[1], 5))
        self.pca.fit(X_scaled)
        self.fitted = True
        self.explained_variance = self.pca.explained_variance_ratio_
        logger.info(f"PCA ajustado. Variância explicada: {self.explained_variance.sum():.3f}")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Projeta dados no espaço PCA e retorna o primeiro componente (score principal).
        """
        if not self.fitted:
            raise ValueError("PCA não ajustado.")
        X_scaled = self.scaler.transform(X)
        scores = self.pca.transform(X_scaled)
        # Usa o primeiro componente como score (se houver mais de um, pode combinar)
        # Para um índice, pegamos o primeiro PC (sinal pode ser invertido, mas ajustamos)
        score = scores[:, 0]
        # Se correlação com a média for negativa, inverte
        if np.corrcoef(score, np.mean(X, axis=1))[0, 1] < 0:
            score = -score
        return score