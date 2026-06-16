# utils/helpers.py
import pandas as pd

def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normaliza uma série entre 0 e 1.
    """
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
    """
    Calcula CAGR (Compound Annual Growth Rate).
    """
    if start_value <= 0 or end_value <= 0 or periods <= 0:
        return 0.0
    return ((end_value / start_value) ** (1 / periods) - 1) * 100

def safe_divide(a: float, b: float) -> float:
    """
    Divisão segura para evitar ZeroDivisionError.
    """
    return a / b if b != 0 else 0
