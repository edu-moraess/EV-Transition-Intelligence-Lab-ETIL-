
# ingestion/loader.py
"""
Módulo para carregamento de dados a partir de arquivos Excel/CSV.
"""
import pandas as pd
from pathlib import Path
from typing import Union, Optional, List
from config.settings import RAW_DATA_DIR, EXPECTED_COLUMNS
from utils.logger import setup_logger
from utils.helpers import validate_columns

logger = setup_logger(__name__)

class DataLoader:
    """
    Carrega e valida dados de fontes externas.
    """

    def __init__(self, raw_dir: Optional[Path] = None):
        """
        Args:
            raw_dir (Path, optional): Diretório onde estão os arquivos brutos.
                                      Default: RAW_DATA_DIR.
        """
        self.raw_dir = raw_dir or RAW_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def load_from_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Carrega dados de um arquivo CSV.

        Args:
            filename (str): Nome do arquivo (ou caminho relativo ao raw_dir).
            **kwargs: Argumentos adicionais para pd.read_csv.

        Returns:
            pd.DataFrame: Dados carregados.
        """
        file_path = self._resolve_path(filename)
        logger.info(f"Carregando CSV: {file_path}")
        df = pd.read_csv(file_path, **kwargs)
        self._validate(df)
        return df

    def load_from_excel(self, filename: str, sheet_name: Union[str, int] = 0, **kwargs) -> pd.DataFrame:
        """
        Carrega dados de um arquivo Excel.

        Args:
            filename (str): Nome do arquivo.
            sheet_name (str|int): Nome ou índice da planilha.
            **kwargs: Argumentos adicionais para pd.read_excel.

        Returns:
            pd.DataFrame: Dados carregados.
        """
        file_path = self._resolve_path(filename)
        logger.info(f"Carregando Excel: {file_path} (sheet={sheet_name})")
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        self._validate(df)
        return df

    def _resolve_path(self, filename: str) -> Path:
        """Resolve caminho absoluto do arquivo."""
        path = Path(filename)
        if not path.is_absolute():
            path = self.raw_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        return path

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Valida a estrutura básica do DataFrame.
        """
        # Verifica colunas obrigatórias (pelo menos country e year e ev_sales)
        required = ["country", "year", "ev_sales"]
        validate_columns(df, required)
        # Verifica se year é inteiro
        if not pd.api.types.is_integer_dtype(df["year"]):
            raise ValueError("Coluna 'year' deve ser inteira.")
        # Verifica se ev_sales é numérico
        if not pd.api.types.is_numeric_dtype(df["ev_sales"]):
            raise ValueError("Coluna 'ev_sales' deve ser numérica.")
        # Adiciona colunas opcionais se não existirem
        for col in ["total_sales", "charging_points", "population", "gdp"]:
            if col not in df.columns:
                df[col] = None
        logger.info(f"Dados carregados com sucesso: {df.shape[0]} registros")