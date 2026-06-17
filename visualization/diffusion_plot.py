# visualization/diffusion_plot.py
"""
Gráficos de curvas de difusão (Bass, Logistic, Gompertz).
"""
import plotly.graph_objects as go
import numpy as np
import pandas as pd

class DiffusionPlot:
    @staticmethod
    def plot_models(t_hist, S_hist, models_dict, t_future=None):
        """
        Plota curvas ajustadas e previsões.
        models_dict: dict com nomes e instâncias de modelos.
        """
        fig = go.Figure()
        # Dados históricos
        fig.add_trace(go.Scatter(x=t_hist, y=S_hist, mode='markers+lines', name='Histórico'))

        for name, model in models_dict.items():
            if model.fitted:
                t_all = np.concatenate([t_hist, t_future]) if t_future is not None else t_hist
                S_pred = model.predict(t_all)
                fig.add_trace(go.Scatter(x=t_all, y=S_pred, mode='lines', name=name))

        fig.update_layout(title="Curvas de Difusão", xaxis_title="Ano (index)", yaxis_title="Adoção Cumulativa")
        return fig