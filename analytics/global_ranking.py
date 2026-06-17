# analytics/global_ranking.py
"""
Ranking global baseado em métricas agregadas.
"""
import pandas as pd
from models.scoring import EVQTI
from utils.logger import setup_logger

logger = setup_logger(__name__)

class GlobalRanking:
    """
    Gera ranking global de países usando EVQTI e outras métricas.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.evqti = EVQTI()
        self.ranking_df = None

    def compute_ranking(self) -> pd.DataFrame:
        """
        Computa ranking EVQTI.
        """
        self.evqti.fit(self.df)
        self.ranking_df = self.evqti.get_ranking()
        return self.ranking_df

    def get_top_n(self, n: int = 10) -> pd.DataFrame:
        """Retorna os n primeiros."""
        if self.ranking_df is None:
            self.compute_ranking()
        return self.ranking_df.head(n)