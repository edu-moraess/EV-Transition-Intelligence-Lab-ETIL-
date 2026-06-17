# ingestion/loader.py
"""
Módulo para carregamento de dados a partir de arquivos Excel/CSV.
Suporta tanto o formato simplificado (country, year, ev_sales, ...)
quanto o formato longo IEA (EV_Data_Explorer_GEVO).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, List
from config.settings import RAW_DATA_DIR, EXPECTED_COLUMNS
from utils.logger import setup_logger
from utils.helpers import validate_columns

logger = setup_logger(__name__)

# Arquivos IEA GEVO em ordem de preferência (mais recente primeiro)
IEA_CANDIDATES = [
    "EV_Data_Explorer_2026_GEVO_clean.csv",
    "EV_Data_Explorer_2025_GEVO_clean.csv",
]


class DataLoader:
    """
    Carrega e valida dados de fontes externas.
    """

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def load_from_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Carrega dados de um arquivo CSV.

        Tenta primeiro o arquivo solicitado. Se estiver vazio ou ausente,
        procura automaticamente por arquivos IEA GEVO no diretório raw.

        Args:
            filename (str): Nome do arquivo (ou caminho relativo ao raw_dir).
            **kwargs: Argumentos adicionais para pd.read_csv.

        Returns:
            pd.DataFrame: Dados no formato padronizado.
        """
        try:
            file_path = self._resolve_path(filename)
            logger.info(f"Carregando CSV: {file_path}")
            df = pd.read_csv(file_path, **kwargs)
            if df.empty:
                raise ValueError("Arquivo CSV vazio.")
            self._validate(df)
            return df
        except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as e:
            logger.warning(f"Não foi possível carregar '{filename}': {e}. "
                           "Procurando arquivos IEA GEVO alternativos...")
            return self._load_iea_gevo()

    def load_from_excel(self, filename: str, sheet_name: Union[str, int] = 0, **kwargs) -> pd.DataFrame:
        """
        Carrega dados de um arquivo Excel.
        """
        file_path = self._resolve_path(filename)
        logger.info(f"Carregando Excel: {file_path} (sheet={sheet_name})")
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        self._validate(df)
        return df

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _resolve_path(self, filename: str) -> Path:
        """Resolve caminho absoluto do arquivo."""
        path = Path(filename)
        if not path.is_absolute():
            path = self.raw_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        return path

    def _validate(self, df: pd.DataFrame) -> None:
        """Valida a estrutura básica do DataFrame."""
        required = ["country", "year", "ev_sales"]
        validate_columns(df, required)
        if not pd.api.types.is_integer_dtype(df["year"]):
            raise ValueError("Coluna 'year' deve ser inteira.")
        if not pd.api.types.is_numeric_dtype(df["ev_sales"]):
            raise ValueError("Coluna 'ev_sales' deve ser numérica.")
        for col in ["total_sales", "charging_points", "population", "gdp"]:
            if col not in df.columns:
                df[col] = None
        logger.info(f"Dados carregados com sucesso: {df.shape[0]} registros")

    # ------------------------------------------------------------------
    # Pipeline IEA GEVO
    # ------------------------------------------------------------------

    def _load_iea_gevo(self) -> pd.DataFrame:
        """
        Carrega e transforma um arquivo IEA GEVO (formato longo) para
        o formato padronizado do ETIL.
        """
        for candidate in IEA_CANDIDATES:
            path = self.raw_dir / candidate
            if path.exists():
                logger.info(f"Usando arquivo IEA GEVO: {path}")
                df_raw = pd.read_csv(path)
                df = self._transform_iea_gevo(df_raw)
                logger.info(f"IEA GEVO transformado: {df.shape[0]} registros de {df['country'].nunique()} países.")
                return df

        raise FileNotFoundError(
            "Nenhum arquivo de dados encontrado. "
            f"Coloque um dos seguintes na pasta data/raw/: "
            f"{', '.join(IEA_CANDIDATES)} ou ev_data.csv"
        )

    def _transform_iea_gevo(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma o formato longo IEA (region_country, parameter, value, ...)
        para o formato wide padronizado (country, year, ev_sales, ...).

        Regras de mapeamento:
          - ev_sales       : parameter='EV sales', powertrain='EV', category='Historical'
          - total_sales    : parameter='EV sales', powertrain IN ('EV','ICEV'), category='Historical', soma por país/ano
          - charging_points: parameter='EV charging points', category='Historical', soma por país/ano
          - ev_stock_share : parameter='EV stock share', powertrain='EV', mode='Cars', category='Historical'
        """
        hist = df_raw[df_raw["category"] == "Historical"].copy()

        # --- ev_sales ---
        ev_sales = (
            hist[
                (hist["parameter"] == "EV sales") &
                (hist["powertrain"] == "EV")
            ]
            .groupby(["region_country", "year"])["value"]
            .sum()
            .reset_index()
            .rename(columns={"region_country": "country", "value": "ev_sales"})
        )

        # --- total_sales (EV + ICEV quando disponível, senão só EV) ---
        total_sales = (
            hist[
                (hist["parameter"] == "EV sales") &
                (hist["powertrain"].isin(["EV", "ICEV"]))
            ]
            .groupby(["region_country", "year"])["value"]
            .sum()
            .reset_index()
            .rename(columns={"region_country": "country", "value": "total_sales"})
        )

        # --- charging_points ---
        charging = (
            hist[hist["parameter"] == "EV charging points"]
            .groupby(["region_country", "year"])["value"]
            .sum()
            .reset_index()
            .rename(columns={"region_country": "country", "value": "charging_points"})
        )

        # --- ev_stock_share (Cars, EV) ---
        share = (
            hist[
                (hist["parameter"] == "EV stock share") &
                (hist["powertrain"] == "EV") &
                (hist["mode"] == "Cars")
            ]
            .groupby(["region_country", "year"])["value"]
            .mean()
            .reset_index()
            .rename(columns={"region_country": "country", "value": "ev_stock_share"})
        )

        # --- Merge ---
        df = ev_sales.copy()
        df = df.merge(total_sales, on=["country", "year"], how="left")
        df = df.merge(charging, on=["country", "year"], how="left")
        df = df.merge(share, on=["country", "year"], how="left")

        # Colunas exigidas pelo pipeline mas não presentes no IEA
        df["population"] = np.nan
        df["gdp"] = np.nan

        # Tipos
        df["year"] = df["year"].astype(int)
        df["ev_sales"] = pd.to_numeric(df["ev_sales"], errors="coerce")

        # Remove regiões agregadas (mantém apenas países individuais)
        # Heurística: nomes de regiões costumam ter vírgulas, parênteses ou palavras-chave
        aggregate_patterns = [
            "World", "Europe", "Economies", "Region", "Asia", "Africa",
            "America", "Pacific", "Rest of", "Other", "Total"
        ]
        mask_aggregate = df["country"].str.contains(
            "|".join(aggregate_patterns), case=False, na=False
        )
        df = df[~mask_aggregate].reset_index(drop=True)

        # Remove linhas sem ev_sales
        df = df.dropna(subset=["ev_sales"])
        df = df[df["ev_sales"] > 0].reset_index(drop=True)

        return df
 