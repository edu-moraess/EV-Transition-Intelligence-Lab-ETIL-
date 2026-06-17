import pandas as pd
from ingestion.loader import DataLoader


class DatasetEngine:
    """
    Engine responsável por resolver qual dataset entra no pipeline:
    - 2025
    - 2026
    - Combined (2025 + 2026)
    """

    def __init__(self, vehicle_mode: str):
        self.loader = DataLoader()
        self.vehicle_mode = vehicle_mode

    def load(self, dataset_mode: str) -> pd.DataFrame:

        # =========================
        # 2025 ONLY
        # =========================
        if dataset_mode == "2025":

            df = self.loader.load_from_csv(
                vehicle_mode=self.vehicle_mode,
                dataset="2025"
            )

            df["year"] = 2025
            return df

        # =========================
        # 2026 ONLY
        # =========================
        if dataset_mode == "2026":

            df = self.loader.load_from_csv(
                vehicle_mode=self.vehicle_mode,
                dataset="2026"
            )

            df["year"] = 2026
            return df

        # =========================
        # COMBINED MODE
        # =========================
        if dataset_mode == "Combined":

            df_2025 = self.loader.load_from_csv(
                vehicle_mode=self.vehicle_mode,
                dataset="2025"
            )
            df_2025["year"] = 2025

            df_2026 = self.loader.load_from_csv(
                vehicle_mode=self.vehicle_mode,
                dataset="2026"
            )
            df_2026["year"] = 2026

            # concat padrão panel data
            df = pd.concat([df_2025, df_2026], ignore_index=True)

            return df

        raise ValueError(
            f"[DatasetEngine] dataset_mode inválido: {dataset_mode}"
        )
