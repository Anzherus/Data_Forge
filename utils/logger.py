"""Настройка технического журнала (отделён от истории сценариев)."""

import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logger(name: str = "dataforge", level: int = logging.INFO) -> logging.Logger:
    """Создаёт/возвращает настроенный logger. Идемпотентно."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(ch)
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Логгер для конкретного модуля, например get_logger('loader')."""
    
    return setup_logger(f"dataforge.{module_name}")