import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """
    Carregador oficial IEA Global EV Outlook.
    Suporta múltiplos datasets (2025 / 2026).
    Versão robusta com fallback de vehicle_mode.
    """

    def __init__(self):
        self.root = Path(__file__).parent.parent

    # ==========================================================
    # LOAD ENTRY POINT
    # ==========================================================
    def load_from_csv(
        self,
        filename: str = None,
        vehicle_mode: str = "Cars",
        dataset: str = "2026"
    ) -> pd.DataFrame:

        # -----------------------------
        # dataset selector
        # -----------------------------
        if filename is None:

            if dataset == "2025":
                filename = "EV_Data_Explorer_2025_GEVO_clean.csv"

            elif dataset == "2026":
                filename = "EV_Data_Explorer_2026_GEVO_clean.csv"

            else:
                raise ValueError(f"Dataset inválido: {dataset}")

        # -----------------------------
        # file search
        # -----------------------------
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
            raise FileNotFoundError(f"Arquivo não encontrado: {filename}")

        logger.info(
            f"IEA bruto ({dataset}): "
            f"{len(df_raw)} linhas | "
            f"{df_raw['region_country'].nunique()} países"
        )

        return self._transform_iea_gevo(df_raw, vehicle_mode)

    # ==========================================================
    # TRANSFORM
    # ==========================================================
    def _transform_iea_gevo(
        self,
        df: pd.DataFrame,
        vehicle_mode: str
    ) -> pd.DataFrame:

        # -----------------------------
        # Historical filter
        # -----------------------------
        if "category" in df.columns:
            df = df[df["category"] == "Historical"].copy()

        # -----------------------------
        # SAFE VEHICLE FILTER (FIX CRÍTICO)
        # -----------------------------
        if vehicle_mode != "Total" and "mode" in df.columns:

            unique_modes = df["mode"].dropna().unique()
            logger.info(f"[MODE DEBUG] disponíveis: {unique_modes}")

            # fallback inteligente
            df["mode_str"] = df["mode"].astype(str)

            if vehicle_mode == "Cars":
                df = df[
                    df["mode_str"].str.contains("Car|Passenger|LDV", na=False)
                ].copy()

            elif vehicle_mode == "2 and 3 wheelers":
                df = df[
                    df["mode_str"].str.contains("2|3|Motorcycle|Two", na=False)
                ].copy()

            else:
                df = df.copy()

        logger.info(
            f"Após filtros: {len(df)} linhas | "
            f"{df['region_country'].nunique()} países"
        )

        # ============================
        # EV SALES
        # ============================
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

        # ============================
        # TOTAL SALES
        # ============================
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

        # ============================
        # CHARGING POINTS
        # ============================
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

        # ============================
        # MARKET SHARE
        # ============================
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

        # ============================
        # MERGE FINAL
        # ============================
        result = (
            ev_sales
            .merge(total_sales, on=["country", "year"], how="left")
            .merge(charging, on=["country", "year"], how="left")
            .merge(share, on=["country", "year"], how="left")
        )

        # ============================
        # SAFE NUMERICS
        # ============================
        result["population"] = np.nan
        result["gdp"] = np.nan

        result["year"] = result["year"].astype(int)

        result["ev_sales"] = pd.to_numeric(
            result["ev_sales"],
            errors="coerce"
        )

        # remove inválidos
        result = result.dropna(subset=["ev_sales"])
        result = result[result["ev_sales"] > 0].reset_index(drop=True)

        logger.info(
            f"ETIL pronto: {len(result)} linhas | "
            f"{result['country'].nunique()} países"
        )

        return result


__all__ = ["DataLoader"]
