# ingestion/dataset_engine.py
"""
Dataset Engine — suporte a IEA GEVO 2025 e 2026.

Schema IEA 2025: powertrain = BEV | PHEV | FCEV  (sem 'EV' agregado)
Schema IEA 2026: powertrain = EV | BEV | PHEV | FCEV | ICEV

A engine normaliza ambos para o schema ETIL.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal
from utils.logger import setup_logger

logger = setup_logger(__name__)

_DATASET_FILES = {
    2025: "EV_Data_Explorer_2025_GEVO_clean.csv",
    2026: "EV_Data_Explorer_2026_GEVO_clean.csv",
}

_PROJECTION_YEARS = {2030, 2035, 2040, 2045, 2050}

# Normalizado em lowercase para comparação robusta (Bug #4 fix v2)
# Lista expandida com agregados do IEA 2026 que estavam faltando
_AGGREGATE_REGIONS_LOWER = {
    # Continentes / macro-regiões
    "world", "europe", "asia pacific", "north america",
    "central and south america", "latin america", "africa",
    "middle east and caspian", "middle east", "caspian",
    "southeast asia", "other asia pacific", "other europe",
    # Grupos econômicos
    "advanced economies", "developing economies excl. china",
    "developing economies", "emerging economies",
    "european union", "eu27", "eu28", "g20", "g7", "oecd",
    # Outros agregados IEA
    "rest of the world", "other",
}

# Powertrains elétricos por schema
_EV_POWERTRAINS_2026 = {"EV"}
_EV_POWERTRAINS_2025 = {"BEV", "PHEV", "FCEV", "EREV"}
_ICE_POWERTRAINS     = {"ICEV"}


class DatasetEngine:

    def __init__(self, raw_dir: Path | None = None):
        self.raw_dir = raw_dir or (Path(__file__).parent.parent / "data" / "raw")

    def load_dataset(
        self,
        year: Literal[2025, 2026],
        vehicle_mode: str = "Cars",
        countries_only: bool = True,
    ) -> pd.DataFrame:
        filename = _DATASET_FILES.get(year)
        if filename is None:
            raise ValueError(f"Dataset {year} não reconhecido.")
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        logger.info(f"Carregando IEA {year}: {path}")
        df_raw = pd.read_csv(path)

        # Detectar schema via powertrains em EV sales
        sales_pts = set(df_raw.loc[df_raw["parameter"] == "EV sales", "powertrain"].unique())
        schema    = "2026" if "EV" in sales_pts else "2025"
        logger.info(f"Schema detectado: IEA {schema}")

        df = self._transform(df_raw, vehicle_mode=vehicle_mode,
                              countries_only=countries_only, schema=schema)
        df["data_source"] = year
        logger.info(f"Dataset {year}: {df.shape[0]} linhas | {df['country'].nunique()} países")
        return df

    def load_all_datasets(
        self,
        vehicle_mode: str = "Cars",
        countries_only: bool = True,
    ) -> pd.DataFrame:
        dfs = []
        for year in [2025, 2026]:
            try:
                df = self.load_dataset(year, vehicle_mode=vehicle_mode,
                                        countries_only=countries_only)
                dfs.append(df)
            except FileNotFoundError:
                logger.warning(f"Dataset {year} não encontrado.")

        if not dfs:
            raise RuntimeError("Nenhum dataset disponível.")

        combined = (
            pd.concat(dfs, ignore_index=True)
            .sort_values("data_source", ascending=False)
            .drop_duplicates(subset=["country", "year"], keep="first")
            .sort_values(["country", "year"])
            .reset_index(drop=True)
        )
        combined["data_source"] = "combined"
        logger.info(f"Combined: {combined.shape[0]} linhas | {combined['country'].nunique()} países")
        return combined

    @staticmethod
    def filter_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
        return df[df["year"] == year].copy()

    # ── Transformação ─────────────────────────────────────────────────────

    def _transform(self, df_raw, vehicle_mode, countries_only, schema):
        # Apenas histórico
        if "category" in df_raw.columns:
            df_raw = df_raw[df_raw["category"] == "Historical"].copy()

        # Remover anos de projeção
        df_raw = df_raw[~df_raw["year"].isin(_PROJECTION_YEARS)].copy()

        # Filtro de modo de transporte — normalização robusta (Bug #1 fix)
        if vehicle_mode != "Total" and "mode" in df_raw.columns:
            mode_norm = vehicle_mode.strip().lower()
            df_raw = df_raw[df_raw["mode"].str.strip().str.lower() == mode_norm].copy()

        if len(df_raw) == 0:
            logger.warning(f"Nenhum dado após filtros (mode={vehicle_mode}).")
            return pd.DataFrame(columns=["country","year","ev_sales","total_sales",
                                          "charging_points","ev_stock_share",
                                          "ev_sales_share","population","gdp","data_source"])

        # Selecionar powertrains EV conforme schema
        ev_pts  = _EV_POWERTRAINS_2026 if schema == "2026" else _EV_POWERTRAINS_2025

        ev_sales    = self._sum_metric(df_raw, "EV sales", ev_pts, "ev_sales")
        total_sales = self._sum_metric(df_raw, "EV sales",
                                        ev_pts | _ICE_POWERTRAINS, "total_sales")
        charging    = self._sum_all(df_raw, "EV charging points", "charging_points")

        # Stock share e Sales share — schema-aware (Bug #3 fix)
        if schema == "2026":
            stock_share = self._mean_metric(df_raw, "EV stock share", {"EV"}, "ev_stock_share")
            sales_share = self._mean_metric(df_raw, "EV sales share", {"EV"}, "ev_sales_share")
        else:
            stock_share = self._sum_metric(df_raw, "EV stock share", _EV_POWERTRAINS_2025, "ev_stock_share")
            sales_share = self._sum_metric(df_raw, "EV sales share", _EV_POWERTRAINS_2025, "ev_sales_share")

        result = (
            ev_sales
            .merge(total_sales, on=["country","year"], how="left")
            .merge(charging,    on=["country","year"], how="left")
            .merge(stock_share, on=["country","year"], how="left")
            .merge(sales_share, on=["country","year"], how="left")
        )

        result["population"] = np.nan
        result["gdp"]        = np.nan
        result["year"]       = result["year"].astype(int)
        result["ev_sales"]   = pd.to_numeric(result["ev_sales"], errors="coerce")

        result = result.dropna(subset=["ev_sales"])
        result = result[result["ev_sales"] > 0].reset_index(drop=True)

        # Filtro de agregados — normalização robusta lowercase (Bug #4 fix v2)
        if countries_only:
            agg_mask = result["country"].str.strip().str.lower().isin(_AGGREGATE_REGIONS_LOWER)
            n_agg = agg_mask.sum()
            if n_agg > 0:
                removed = result[agg_mask]["country"].unique().tolist()
                logger.warning(f"Agregados removidos: {removed}")
            result = result[~agg_mask].reset_index(drop=True)

        return result

    # ── Helpers de extração ───────────────────────────────────────────────

    def _sum_metric(self, df, parameter, powertrains, col):
        mask = (df["parameter"] == parameter) & (df["powertrain"].isin(powertrains))
        return (
            df[mask].groupby(["region_country","year"])["value"]
            .sum().reset_index()
            .rename(columns={"region_country":"country","value":col})
        )

    def _mean_metric(self, df, parameter, powertrains, col):
        mask = (df["parameter"] == parameter) & (df["powertrain"].isin(powertrains))
        return (
            df[mask].groupby(["region_country","year"])["value"]
            .mean().reset_index()
            .rename(columns={"region_country":"country","value":col})
        )

    def _sum_all(self, df, parameter, col):
        mask = df["parameter"] == parameter
        return (
            df[mask].groupby(["region_country","year"])["value"]
            .sum().reset_index()
            .rename(columns={"region_country":"country","value":col})
        )


__all__ = ["DatasetEngine"]
