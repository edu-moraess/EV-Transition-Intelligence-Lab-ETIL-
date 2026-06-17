# ingestion/loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    """Carregador robusto para dados IEA GEVO"""

    def __init__(self):
        self.root = Path(__file__).parent.parent

    def load_from_csv(self, filename: str = "ev_data.csv") -> pd.DataFrame:
        """Tenta carregar arquivo processado ou IEA bruto"""
        paths_to_try = [
            self.root / "data/processed" / filename,
            self.root / "data/raw" / filename,
            Path(filename),
            self.root / "EV_Data_Explorer_2026_GEVO_clean.csv",
            self.root / "attachments/EV_Data_Explorer_2026_GEVO_clean.csv"
        ]

        for path in paths_to_try:
            if path.exists():
                logger.info(f"Carregando: {path.name}")
                df = pd.read_csv(path)
                return self._transform_iea_gevo(df)

        logger.warning("Nenhum arquivo CSV encontrado. Usando dummy data.")
        return self._generate_dummy_data()

    def _transform_iea_gevo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma formato longo IEA para wide"""
        if 'region_country' not in df.columns:
            return df

        # Renomeia para consistência
        df = df.rename(columns={'region_country': 'country'})

        # Pivot principal
        pivot = df.pivot_table(
            index=['country', 'year'],
            columns=['parameter', 'mode', 'powertrain'],
            values='value',
            aggfunc='first'
        ).reset_index()

        # Colunas principais
        col_map = {
            ('EV sales', 'Cars', 'EV'): 'ev_sales',
            ('EV stock', 'Cars', 'EV'): 'ev_stock',
            ('EV charging points', 'EVSE', 'Publicly available fast'): 'charging_points_fast',
        }

        for old, new in col_map.items():
            if old in pivot.columns:
                pivot[new] = pivot[old]

        # Total sales proxy quando não existe
        if 'ev_sales' in pivot.columns and 'total_sales' not in pivot.columns:
            pivot['total_sales'] = pivot['ev_sales'] * 5  # proxy conservador

        pivot = pivot.fillna(0)
        return pivot

    def _generate_dummy_data(self):
        """Dados dummy de fallback"""
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


# Para importação direta
__all__ = ["DataLoader"]