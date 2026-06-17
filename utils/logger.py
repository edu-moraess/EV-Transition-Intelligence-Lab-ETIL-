# utils/logger.py
"""
Configuração de logging para o sistema.
"""
import logging
import sys
from pathlib import Path

from config.settings import LOG_FILE, LOG_LEVEL

def setup_logger(name: str = "etil") -> logging.Logger:
    """
    Cria e configura um logger com saída para console e arquivo.

    Args:
        name (str): Nome do logger.

    Returns:
        logging.Logger: Instância configurada.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Evita duplicação de handlers
    if logger.hasHandlers():
        return logger

    # Formato
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para arquivo
    log_file = Path(LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger