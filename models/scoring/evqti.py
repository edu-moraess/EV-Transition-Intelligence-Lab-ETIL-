# models/scoring/evqti.py
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)

class EVQTI:
    """
    EV Quant Transition Index.
    Combina métricas de crescimento, penetração, momentum e infraestrutura.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute_scores(self) -> pd.DataFrame:
        scores = self.df.groupby("country").agg({
            "cagr": "mean",
            "penetration": "mean",
            "momentum_3y": "mean",
            "ev_per_charger": "mean",
            "ev_per_capita": "mean"
        }).fillna(0)

        # Normalização simples
        for col in scores.columns:
            scores[col] = (scores[col] - scores[col].min()) / (scores[col].max() - scores[col].min() + 1e-6)

        # Score final
        scores["EVQTI"] = scores.mean(axis=1) * 100
        logger.info("EVQTI calculado.")
        return scores.reset_index()
