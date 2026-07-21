import sys
import time
import logging
import functools
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Define operational logging paths
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "omnibrain_backend.log"

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("omnibrain")
    logger.setLevel(log_level)
    
    if logger.handlers:
        return logger

    # Console Output Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] -> %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Rotating Persistent Log File Handler
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "origin": "%(filename)s:%(lineno)d", "message": "%(message)s"}'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

# Global logger instance
logger = setup_logging()

# Performance Telemetry Decorator
def time_execution(stage_name: str):
    """Decorator to log processing duration of functions (e.g., parsing, vectorization)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                logger.info(f"PERFORMANCE: [{stage_name}] completed in {duration:.4f}s")
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(f"EXCEPTION: [{stage_name}] failed after {duration:.4f}s with error: {str(e)}")
                raise e
        return wrapper
    return decorator