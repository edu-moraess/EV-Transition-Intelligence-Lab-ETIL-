# models/scoring/pca_score.py
import pandas as pd
from sklearn.decomposition import PCA
from utils.logger import setup_logger

logger = setup_logger(__name__)

class PCAScorer:
    """
    Redução de dimensionalidade via PCA.
    """

    def __init__(self, n_components: int = 2):
        self.pca = PCA(n_components=n_components)
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        X = df[features].fillna(0).values
        components = self.pca.fit_transform(X)
        self.fitted = True
        logger.info("PCA ajustado.")
        return pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(components.shape[1])])
