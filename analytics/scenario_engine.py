# analytics/scenario_engine.py
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ScenarioEngine:
    """
    Motor de cenários 'what-if' para EV.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def apply_growth_shock(self, country: str, shock: float) -> pd.DataFrame:
        df_country = self.df[self.df["country"] == country].copy()
        df_country["ev_sales_shocked"] = df_country["ev_sales"] * (1 + shock)
        logger.info(f"Choque de crescimento {shock:.2f} aplicado em {country}.")
        return df_country

    def infrastructure_boost(self, country: str, boost: float) -> pd.DataFrame:
        df_country = self.df[self.df["country"] == country].copy()
        if "ev_per_charger" in df_country:
            df_country["ev_per_charger_boosted"] = df_country["ev_per_charger"] * (1 - boost)
        logger.info(f"Boost de infraestrutura {boost:.2f} aplicado em {country}.")
        return df_country
