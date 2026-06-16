# analytics/global_ranking.py
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)

class GlobalRanking:
    """
    Ranking global de países por métricas de EV.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def rank_by_metric(self, metric: str, top_n: int = 10) -> pd.DataFrame:
        if metric not in self.df.columns:
            raise ValueError(f"Métrica {metric} não encontrada.")
        ranking = self.df.groupby("country")[metric].mean().sort_values(ascending=False).head(top_n)
        logger.info(f"Ranking global gerado por {metric}.")
        return ranking.reset_index()
