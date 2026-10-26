"""Logging configuration module.

Provides a unified logger with formatted console and rotating file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import LOGS_DIR


def get_logger(name: str = "bank_loan", log_file: Optional[str] = "pipeline.log") -> logging.Logger:
    """Configures and returns a thread-safe logger.

    Args:
        name: Name of the logger instance.
        log_file: Optional file name inside reports/logs/.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
