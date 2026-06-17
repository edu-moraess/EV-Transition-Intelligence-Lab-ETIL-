# models/regimes/hmm.py
"""
Detecção de regimes usando Hidden Markov Models.
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class HMMRegimeDetector:
    """
    Aplica HMM para identificar estados ocultos (regimes) em séries temporais de adoção.
    """
    REGIME_NAMES = {0: "Emerging", 1: "Expansion", 2: "Acceleration", 3: "Maturity"}

    def __init__(self, n_states: int = 4, random_seed: int = 42):
        self.n_states = n_states
        self.random_seed = random_seed
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        """
        Ajusta o HMM aos dados.

        Args:
            X (np.ndarray): Array 2D de features (n_samples, n_features).
        """
        # Normaliza
        X_scaled = self.scaler.fit_transform(X)
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=self.random_seed,
            verbose=False
        )
        self.model.fit(X_scaled)
        self.fitted = True
        logger.info("HMM ajustado com sucesso.")

    def predict_regime(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediz os estados ocultos e suas probabilidades.

        Args:
            X (np.ndarray): Dados de entrada.

        Returns:
            Tuple[np.ndarray, np.ndarray]: estados previstos, probabilidades.
        """
        if not self.fitted:
            raise ValueError("Modelo não ajustado.")
        X_scaled = self.scaler.transform(X)
        states = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)
        return states, probs

    def get_regime_name(self, state_id: int) -> str:
        """Retorna o nome descritivo do regime."""
        return self.REGIME_NAMES.get(state_id, f"Regime {state_id}")

    def get_transition_matrix(self) -> np.ndarray:
        """Retorna a matriz de transição."""
        if self.model is None:
            raise ValueError("Modelo não inicializado.")
        return self.model.transmat_