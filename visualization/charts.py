# visualization/charts.py
"""
Funções genéricas para criação de gráficos com matplotlib e plotly.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List

class Charts:
    @staticmethod
    def line_chart(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, title: str = ""):
        """Gráfico de linha com plotly."""
        fig = px.line(df, x=x, y=y, color=hue, title=title)
        return fig

    @staticmethod
    def bar_chart(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, title: str = ""):
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        return fig

    @staticmethod
    def scatter_plot(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, size: Optional[str] = None,
                     title: str = ""):
        fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title)
        return fig