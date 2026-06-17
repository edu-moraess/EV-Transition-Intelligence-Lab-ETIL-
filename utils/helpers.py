# utils/helpers.py
"""
Funções auxiliares gerais.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from sklearn.preprocessing import StandardScaler

def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    Verifica se todas as colunas obrigatórias estão presentes.

    Args:
        df (pd.DataFrame): DataFrame a ser validado.
        required_cols (List[str]): Lista de colunas obrigatórias.

    Returns:
        bool: True se todas existirem.
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
    return True

def standardize_series(series: pd.Series) -> pd.Series:
    """
    Padroniza uma série (z-score).

    Args:
        series (pd.Series): Série de dados.

    Returns:
        pd.Series: Série padronizada.
    """
    return (series - series.mean()) / series.std() if series.std() != 0 else series

def growth_rate(series: pd.Series) -> pd.Series:
    """
    Calcula a taxa de crescimento anual (percentual) de uma série.

    Args:
        series (pd.Series): Série temporal (ordenada).

    Returns:
        pd.Series: Taxa de crescimento ano a ano (em %).
    """
    return series.pct_change() * 100

def cagr(series: pd.Series) -> float:
    """
    Calcula a Taxa de Crescimento Anual Composta (CAGR).

    Args:
        series (pd.Series): Série temporal (ordenada).

    Returns:
        float: CAGR em porcentagem.
    """
    if len(series) < 2:
        return np.nan
    n = len(series) - 1
    return ( (series.iloc[-1] / series.iloc[0]) ** (1/n) - 1 ) * 100

def rolling_momentum(series: pd.Series, window: int = 3) -> pd.Series:
    """
    Calcula o momentum como a diferença entre a média recente e a média anterior.

    Args:
        series (pd.Series): Série temporal.
        window (int): Tamanho da janela.

    Returns:
        pd.Series: Momentum.
    """
    if len(series) < 2 * window:
        return pd.Series(index=series.index, dtype=float)
    recent = series.rolling(window).mean()
    previous = series.shift(window).rolling(window).mean()
    return (recent - previous) / previous.abs() * 100

def sigmoid(x: np.ndarray, a: float = 1, b: float = 0) -> np.ndarray:
    """
    Função sigmoide para suavização.
    """
    return 1 / (1 + np.exp(-a * (x - b)))