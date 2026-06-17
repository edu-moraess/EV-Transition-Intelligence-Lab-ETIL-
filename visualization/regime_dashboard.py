# visualization/regime_dashboard.py
"""
Dashboard para visualização de regimes.
"""
import plotly.graph_objects as go
import numpy as np

class RegimeDashboard:
    @staticmethod
    def plot_regimes(series, states, probs, regime_names):
        fig = go.Figure()
        # Série principal
        fig.add_trace(go.Scatter(x=np.arange(len(series)), y=series, mode='lines', name='Série'))
        # Regimes como cores de fundo? Usamos áreas
        for i, state in enumerate(np.unique(states)):
            mask = states == state
            # Destaca regiões
            fig.add_trace(go.Scatter(
                x=np.arange(len(series))[mask],
                y=series[mask],
                mode='markers',
                marker=dict(color=f'rgba({(state*50)%255}, 100, 200, 0.5)'),
                name=regime_names[state]
            ))
        fig.update_layout(title="Detecção de Regimes")
        return fig