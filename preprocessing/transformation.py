# preprocessing/transformation.py
"""
Transformações de dados: log, diferenças, normalizações.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataTransformer:
    """
    Aplica transformações matemáticas às séries temporais.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def add_log_transform(self, columns: List[str]) -> pd.DataFrame:
        """Adiciona colunas com log natural das variáveis."""
        for col in columns:
            if col in self.df.columns:
                new_col = f"{col}_log"
                self.df[new_col] = np.log(self.df[col].clip(lower=1e-6))
        return self.df

    def add_diff(self, columns: List[str], periods: int = 1) -> pd.DataFrame:
        """Adiciona diferenças (crescimento) para séries."""
        for col in columns:
            if col in self.df.columns:
                new_col = f"{col}_diff{periods}"
                self.df[new_col] = self.df.groupby("country")[col].diff(periods)
        return self.df

    def add_rolling_stats(self, columns: List[str], window: int = 3) -> pd.DataFrame:
        """Adiciona médias e desvios móveis."""
        for col in columns:
            if col in self.df.columns:
                self.df[f"{col}_rollmean_{window}"] = self.df.groupby("country")[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                self.df[f"{col}_rollstd_{window}"] = self.df.groupby("country")[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
        return self.df