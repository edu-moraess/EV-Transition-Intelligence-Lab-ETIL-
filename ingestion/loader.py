# ingestion/loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    """Carregador robusto com separação por tipo de veículo"""

    def __init__(self):
        self.root = Path(__file__).parent.parent

    def load_from_csv(self, filename: str = "EV_Data_Explorer_2026_GEVO_clean.csv", 
                     vehicle_mode: str = "Cars") -> pd.DataFrame:
        
        # Tenta carregar o arquivo principal
        paths = [
            self.root / "data/raw" / filename,
            Path(filename),
            self.root / filename
        ]
        
        df_raw = None
        for p in paths:
            if p.exists():
                logger.info(f"✅ Carregando {p.name}")
                df_raw = pd.read_csv(p)
                break
                
        if df_raw is None:
            logger.warning("Arquivo não encontrado. Usando dummy.")
            return self._generate_dummy_data()

        # === SEPARAÇÃO POR TIPO DE VEÍCULO ===
        if 'mode' in df_raw.columns:
            if vehicle_mode == "Cars":
                df_raw = df_raw[df_raw['mode'] == "Cars"].copy()
                logger.info("🔹 Filtrado apenas para Cars (Passageiros)")
            elif vehicle_mode == "2 and 3 wheelers":
                df_raw = df_raw[df_raw['mode'] == "2 and 3 wheelers"].copy()
                logger.info("🔹 Filtrado para 2 and 3 Wheelers")
            else:
                logger.info("🔹 Usando todos os tipos de veículo (Total)")

        return self._transform_iea_gevo(df_raw)

    def _transform_iea_gevo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma o formato longo do IEA"""
        if 'region_country' not in df.columns:
            return df

        df = df.rename(columns={'region_country': 'country'}).copy()

        # Pivot principal
        pivot = df.pivot_table(
            index=['country', 'year'],
            columns=['parameter', 'mode', 'powertrain'],
            values='value',
            aggfunc='first'
        ).reset_index()

        # Colunas principais
        if ('EV sales', 'Cars', 'EV') in pivot.columns:
            pivot['ev_sales'] = pivot[('EV sales', 'Cars', 'EV')]
        elif 'ev_sales' not in pivot.columns:
            pivot['ev_sales'] = 0

        if 'ev_sales' in pivot.columns and 'total_sales' not in pivot.columns:
            pivot['total_sales'] = pivot['ev_sales'] * 4  # proxy realista

        pivot = pivot.fillna(0)
        return pivot

    def _generate_dummy_data(self):
        """Dummy data de fallback"""
        np.random.seed(42)
        countries = ["USA","China","Germany","France","UK","Norway","Japan","India","Brazil"]
        years = list(range(2015, 2026))
        data = []
        for c in countries:
            base = 1000 if c in ["USA","China"] else 100
            for y in years:
                ev = base * (1.2 ** (y-2015)) * (0.9 + 0.2*np.random.rand())
                total = ev * (5 + 10*np.random.rand())
                data.append([c, y, ev, total, ev*0.02, 50e6 if c in ["USA","China"] else 10e6, 2e12 if c in ["USA","China"] else 0.5e12])
        return pd.DataFrame(data, columns=["country","year","ev_sales","total_sales","charging_points","population","gdp"])


__all__ = ["DataLoader"]