# visualization/forecast_plot.py
import plotly.graph_objects as go

def plot_forecast(years, actual, pred, std=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=actual, mode="markers", name="Dados"))
    fig.add_trace(go.Scatter(x=years, y=pred, mode="lines", name="Forecast"))
    if std is not None:
        fig.add_trace(go.Scatter(x=years, y=pred + 2*std, line=dict(dash="dot"), name="+2σ"))
        fig.add_trace(go.Scatter(x=years, y=pred - 2*std, line=dict(dash="dot"), name="-2σ"))
    fig.update_layout(title="Previsão EV")
    return fig
