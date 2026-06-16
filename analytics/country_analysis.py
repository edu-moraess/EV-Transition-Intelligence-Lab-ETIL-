# analytics/country_analysis.py
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CountryAnalysis:
    """
    Análises específicas por país.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary(self, country: str) -> dict:
        df_country = self.df[self.df["country"] == country]
        summary = {
            "total_ev_sales": df_country["ev_sales"].sum(),
            "avg_cagr": df_country["cagr"].mean(),
            "avg_penetration": df_country["penetration"].mean() if "penetration" in df_country else None,
            "ev_per_capita": df_country["ev_per_capita"].mean() if "ev_per_capita" in df_country else None,
        }
        logger.info(f"Resumo gerado para {country}.")
        return summary

    def top_years(self, country: str, n: int = 5) -> pd.DataFrame:
        df_country = self.df[self.df["country"] == country]
        return df_country.sort_values("ev_sales", ascending=False).head(n)
