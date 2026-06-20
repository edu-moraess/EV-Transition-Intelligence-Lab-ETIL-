# config/settings.py
"""
Global project configuration.

Centralised here so that paths and model hyperparameters are defined once
and imported everywhere, instead of being duplicated (and potentially
diverging) across ingestion/loader.py, models/regimes/hmm.py,
models/forecasting/monte_carlo.py, etc.
"""
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR            = Path(__file__).parent.parent
DATA_DIR             = ROOT_DIR / "data"
RAW_DATA_DIR         = DATA_DIR / "raw"
PROCESSED_DATA_DIR   = DATA_DIR / "processed"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = ROOT_DIR / "logs" / "etil.log"

# ── Model hyperparameters ──────────────────────────────────────────────────────
BASS_MAX_ITER     = 1000
BASS_TOL          = 1e-6
HMM_N_STATES      = 4          # default regime count for GaussianMixture proxy
HMM_RANDOM_SEED   = 42
GP_N_SAMPLES      = 100
MC_N_SIMULATIONS  = 2000       # matches MonteCarloSimulator default in Forecast Lab
FORECAST_HORIZON  = 10         # years, default in Forecast Lab slider

# ── EVQTI ─────────────────────────────────────────────────────────────────────
EVQTI_WINSORIZE_FEATURES = (0.01, 0.99)
EVQTI_WINSORIZE_TARGET   = (0.05, 0.95)
EVQTI_BACKTEST_TRAIN_YEARS = 6
EVQTI_BACKTEST_TEST_YEARS  = 1

# ── IEA dataset schema ──────────────────────────────────────────────────────────
# Reflects the real primary key enforced by ingestion/aggregator.py
# (country × category × year × parameter × mode × powertrain), not a
# simplified wide-format column list — kept here for documentation only,
# loader.py validates dynamically rather than against this static list.
IEA_RAW_REQUIRED_COLUMNS = [
    "region_country", "category", "parameter",
    "mode", "powertrain", "year", "value",
]

IEA_PANEL_COLUMNS = [
    "country", "category", "year", "mode",
    "ev_sales", "ev_stock", "ev_sales_share",
    "ev_stock_share", "charging_points", "data_source",
]
