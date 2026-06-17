# preprocessing/cleaning.py
"""
Módulo para limpeza de dados: valores ausentes, duplicatas, padronização de países.
"""
import pandas as pd
import numpy as np
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataCleaner:
    """
    Realiza limpeza e padronização do dataset.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df (pd.DataFrame): DataFrame bruto.
        """
        self.df = df.copy()

    def clean(self) -> pd.DataFrame:
        """
        Executa todas as etapas de limpeza.

        Returns:
            pd.DataFrame: DataFrame limpo.
        """
        self._remove_duplicates()
        self._handle_missing()
        self._standardize_countries()
        self._convert_types()
        self._validate_temporal()
        return self.df

    def _remove_duplicates(self) -> None:
        """Remove linhas duplicadas com base em país e ano."""
        initial = len(self.df)
        self.df.drop_duplicates(subset=["country", "year"], inplace=True)
        logger.info(f"Removidas {initial - len(self.df)} duplicatas.")

    def _handle_missing(self) -> None:
        """Trata valores ausentes (interpolação linear para ev_sales)."""
        # Interpola ev_sales por país
        self.df["ev_sales"] = self.df.groupby("country")["ev_sales"].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        # Para colunas opcionais, podemos preencher com 0 ou mediana, mas aqui mantemos None
        for col in ["total_sales", "charging_points", "population", "gdp"]:
            if col in self.df.columns and self.df[col].isnull().any():
                # Preenche com mediana do país se possível
                self.df[col] = self.df.groupby("country")[col].transform(
                    lambda x: x.fillna(x.median() if not x.isna().all() else 0)
                )
        logger.info("Valores ausentes tratados.")

    def _standardize_countries(self) -> None:
        """Padroniza nomes de países (capitaliza, remove espaços)."""
        self.df["country"] = self.df["country"].str.strip().str.title()
        # Mapeamento de sinônimos (exemplo)
        mapping = {
            "United States": "United States",
            "Usa": "United States",
            "U.S.": "United States",
            "Uk": "United Kingdom",
            "U.K.": "United Kingdom",
        }
        self.df["country"] = self.df["country"].replace(mapping)
        logger.info("Nomes de países padronizados.")

    def _convert_types(self) -> None:
        """Converte tipos de colunas."""
        self.df["year"] = self.df["year"].astype(int)
        for col in ["ev_sales", "total_sales", "charging_points", "population", "gdp"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def _validate_temporal(self) -> None:
        """Valida consistência temporal: anos únicos por país, sem gaps grandes."""
        # Verifica se há anos faltantes (gap > 1) -> log warning
        df_grouped = self.df.groupby("country")["year"].apply(list)
        for country, years in df_grouped.items():
            years_sorted = sorted(years)
            gaps = np.diff(years_sorted)
            if any(gaps > 1):
                logger.warning(f"País {country} apresenta gaps nos anos: {years_sorted}")
        logger.info("Validação temporal concluída.")