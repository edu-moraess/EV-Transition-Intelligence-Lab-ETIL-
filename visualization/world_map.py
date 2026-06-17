# visualization/world_map.py
"""
Mapa global com indicadores.
"""
import plotly.express as px
import pandas as pd

class WorldMap:
    @staticmethod
    def choropleth(df: pd.DataFrame, locations_col: str = "country", value_col: str = "ev_sales",
                   title: str = "Vendas de EV por País"):
        fig = px.choropleth(
            df,
            locations=locations_col,
            locationmode="country names",
            color=value_col,
            hover_name=locations_col,
            title=title,
            color_continuous_scale=px.colors.sequential.Plasma
        )
        return fig