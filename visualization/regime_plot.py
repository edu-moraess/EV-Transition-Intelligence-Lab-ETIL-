# visualization/regime_plot.py
import plotly.express as px

def plot_regimes(df, x="year", y="ev_sales", regime_col="regime"):
    fig = px.scatter(df, x=x, y=y, color=regime_col, title="Regimes de EV")
    return fig
