# visualization/diffusion_plot.py
import plotly.graph_objects as go

def plot_diffusion(years, actual, bass, logistic, gompertz):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=actual, mode="markers", name="Dados"))
    fig.add_trace(go.Scatter(x=years, y=bass, mode="lines", name="Bass"))
    fig.add_trace(go.Scatter(x=years, y=logistic, mode="lines", name="Logístico"))
    fig.add_trace(go.Scatter(x=years, y=gompertz, mode="lines", name="Gompertz"))
    fig.update_layout(title="Modelos de Difusão")
    return fig
