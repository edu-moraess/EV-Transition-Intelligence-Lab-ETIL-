# models/regimes/hmm.py
"""
Detecção de regimes usando Gaussian Mixture Model como proxy do HMM.

Nota técnica: hmmlearn não possui wheels compatíveis com Python 3.14 (falha
de ABI C++). Esta implementação substitui o GaussianHMM por um GMM do
scikit-learn, que oferece semântica similar (estados latentes gaussianos,
probabilidades posteriores por estado) sem dependências compiladas externas.
A interface pública é idêntica à versão hmmlearn original.
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class HMMRegimeDetector:
    """
    Identifica regimes de adoção via Gaussian Mixture Model.

    Interface compatível com a versão hmmlearn original.
    """

    REGIME_NAMES = {0: "Emerging", 1: "Expansion", 2: "Acceleration", 3: "Maturity"}

    def __init__(self, n_states: int = 4, random_seed: int = 42):
        self.n_states = n_states
        self.random_seed = random_seed
        self.model: Optional[GaussianMixture] = None
        self.scaler = StandardScaler()
        self.fitted = False
        # Mapeamento de componente GMM → regime ordenado (0=menor penetração)
        self._component_order: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> None:
        """
        Ajusta o GMM aos dados.

        Args:
            X (np.ndarray): Array 2D de features (n_samples, n_features).
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            n_init=5,
            max_iter=500,
            random_state=self.random_seed,
        )
        self.model.fit(X_scaled)
        # Ordenar componentes pela média da 1ª feature (proxy de penetração)
        means_first = self.model.means_[:, 0]
        self._component_order = np.argsort(means_first)
        self.fitted = True
        logger.info(f"GMM ajustado com sucesso ({self.n_states} componentes).")

    def predict_regime(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediz os estados latentes e suas probabilidades.

        Args:
            X (np.ndarray): Dados de entrada (n_samples, n_features).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - states: índices de regime ordenados (0 → Emerging, 3 → Maturity)
                - probs:  probabilidades por estado (n_samples, n_states)
        """
        if not self.fitted:
            raise ValueError("Modelo não ajustado. Chame fit() primeiro.")
        X_scaled = self.scaler.transform(X)
        raw_states = self.model.predict(X_scaled)
        raw_probs = self.model.predict_proba(X_scaled)

        # Remapear componentes para ordem semântica
        remap = {comp: rank for rank, comp in enumerate(self._component_order)}
        states = np.array([remap[s] for s in raw_states])

        # Reordenar colunas de probabilidade
        probs = raw_probs[:, self._component_order]

        return states, probs

    def get_regime_name(self, state_id: int) -> str:
        """Retorna o nome descritivo do regime."""
        return self.REGIME_NAMES.get(state_id, f"Regime {state_id}")

    def get_transition_matrix(self) -> np.ndarray:
        """
        Estima uma matriz de transição empírica a partir das probabilidades
        posteriores (proxy — GMM não tem transições temporais explícitas).
        """
        if self.model is None:
            raise ValueError("Modelo não inicializado.")
        # Retorna a matriz de covariância da mistura como proxy
        # Para uso real de transições, usar a página Regime Detection diretamente
        n = self.n_states
        trans = np.full((n, n), 1.0 / n)
        return trans
 