# models/forecasting/monte_carlo.py
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MonteCarloSimulator:
    """
    Simulações de Monte Carlo para séries de EV.
    """

    def __init__(self, series: np.ndarray):
        self.series = series

    def run(self, n_simulations: int = 1000, horizon: int = 10) -> np.ndarray:
        last_value = self.series[-1]
        growth_rates = np.diff(self.series) / self.series[:-1]
        mu, sigma = np.mean(growth_rates), np.std(growth_rates)

        simulations = np.zeros((n_simulations, horizon))
        for i in range(n_simulations):
            values = [last_value]
            for _ in range(horizon):
                shock = np.random.normal(mu, sigma)
                values.append(values[-1] * (1 + shock))
            simulations[i, :] = values[1:]
        logger.info(f"Monte Carlo rodado com {n_simulations} simulações.")
        return simulations
