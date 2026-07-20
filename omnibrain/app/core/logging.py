import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Define operational logging paths
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "omnibrain_backend.log"

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("omnibrain")
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers if re-initialized
    if logger.handlers:
        return logger

    # 1. Standard Console Output Formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] -> %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 2. Persistent Rotating File Output Configuration
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "origin": "%(filename)s:%(lineno)d", "message": "%(message)s"}'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

# Instantiate global pipeline logger
logger = setup_logging()