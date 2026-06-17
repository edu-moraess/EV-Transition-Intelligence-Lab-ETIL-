# visualization/forecast_dashboard.py
"""
Dashboard de previsão com incertezas.
"""
import plotly.graph_objects as go
import numpy as np

class ForecastDashboard:
    @staticmethod
    def plot_forecast(t_hist, y_hist, t_future, mean, lower, upper, scenario_label="Previsão"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_hist, y=y_hist, mode='lines+markers', name='Histórico'))
        fig.add_trace(go.Scatter(x=t_future, y=mean, mode='lines', name=scenario_label))
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_future, t_future[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalo de Confiança'
        ))
        fig.update_layout(title="Previsão e Incerteza")
        return fig