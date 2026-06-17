# config/settings.py
"""
Configurações globais do projeto.
"""
import os
from pathlib import Path

# Diretório raiz do projeto
ROOT_DIR = Path(__file__).parent.parent

# Diretórios de dados
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Diretório de saída
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Configurações de logs
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = ROOT_DIR / "logs" / "etil.log"

# Configuração de modelos
BASS_MAX_ITER = 1000
BASS_TOL = 1e-6
HMM_N_STATES = 4
HMM_RANDOM_SEED = 42
GP_N_SAMPLES = 100
MC_N_SIMULATIONS = 5000
FORECAST_HORIZON = 10  # anos

# Colunas esperadas no dataset
EXPECTED_COLUMNS = [
    "country",
    "year",
    "ev_sales",
    "total_sales",
    "charging_points",
    "population",
    "gdp",
]