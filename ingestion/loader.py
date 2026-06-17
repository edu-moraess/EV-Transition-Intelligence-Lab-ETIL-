import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """
    Carregador oficial IEA Global EV Outlook.
    Converte formato longo GEVO para formato ETIL.
    """

    def __init__(self):
        self.root = Path(__file__).parent.parent


    def load_from_csv(
        self,
        filename: str = "EV_Data_Explorer_2026_GEVO_clean.csv",
        vehicle_mode: str = "Cars"
    ) -> pd.DataFrame:

        paths = [
            self.root / "data" / "raw" / filename,
            self.root / filename,
            Path(filename)
        ]

        df_raw = None

        for path in paths:
            if path.exists():
                logger.info(f"Carregando {path}")
                df_raw = pd.read_csv(path)
                break

        if df_raw is None:
            raise FileNotFoundError(
                "Arquivo IEA não encontrado."
            )

        logger.info(
            f"IEA bruto: {len(df_raw)} linhas | "
            f"{df_raw['region_country'].nunique()} países/regiões"
        )

        return self._transform_iea_gevo(df_raw, vehicle_mode)


    def _transform_iea_gevo(
        self,
        df: pd.DataFrame,
        vehicle_mode: str
    ) -> pd.DataFrame:

        # Apenas histórico
        if "category" in df.columns:
            df = df[df["category"] == "Historical"].copy()

        # Filtro por veículo (original simples)
        if (
            vehicle_mode != "Total"
            and "mode" in df.columns
        ):
            df = df[
                df["mode"] == vehicle_mode
            ].copy()

        logger.info(
            f"Após filtros: {len(df)} linhas | "
            f"{df['region_country'].nunique()} países"
        )

        # EV sales
        ev_sales = (
            df[
                (df["parameter"] == "EV sales")
                &
                (df["powertrain"] == "EV")
            ]
            .groupby(["region_country", "year"])["value"]
            .sum()
            .reset_index()
            .rename(columns={
                "region_country": "country",
                "value": "ev_sales"
            })
        )

        # Total sales
        total_sales = (
            df[
                (df["parameter"] == "EV sales")
                &
                (df["powertrain"].isin(["EV", "ICEV"]))
            ]
            .groupby(["region_country", "year"])["value"]
            .sum()
            .reset_index()
            .rename(columns={
                "region_country": "country",
                "value": "total_sales"
            })
        )

        # Charging points
        charging = (
            df[df["parameter"] == "EV charging points"]
            .groupby(["region_country", "year"])["value"]
            .sum()
            .reset_index()
            .rename(columns={
                "region_country": "country",
                "value": "charging_points"
            })
        )

        # EV stock share
        share = (
            df[df["parameter"] == "EV stock share"]
            .groupby(["region_country", "year"])["value"]
            .mean()
            .reset_index()
            .rename(columns={
                "region_country": "country",
                "value": "ev_stock_share"
            })
        )

        # Merge final
        result = (
            ev_sales
            .merge(total_sales, on=["country", "year"], how="left")
            .merge(charging, on=["country", "year"], how="left")
            .merge(share, on=["country", "year"], how="left")
        )

        # Campos auxiliares
        result["population"] = np.nan
        result["gdp"] = np.nan

        result["year"] = result["year"].astype(int)

        result["ev_sales"] = pd.to_numeric(
            result["ev_sales"],
            errors="coerce"
        )

        # Remove inválidos
        result = result.dropna(subset=["ev_sales"])
        result = result[result["ev_sales"] > 0].reset_index(drop=True)

        logger.info(
            f"ETIL pronto: {len(result)} linhas | "
            f"{result['country'].nunique()} países"
        )

        return result


__all__ = ["DataLoader"]
