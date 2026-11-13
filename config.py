"""Конфигурация проекта: загрузка/сохранение настроек (JSON)."""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from utils.exceptions import ConfigError
from utils.logger import get_logger

log = get_logger("config")

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


@dataclass
class AppConfig:
    """Настройки приложения (сохраняются между запусками)."""

    data_dir: str = "data"
    processed_dir: str = "processed"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    db_path: str = "db/dataforge.db"

    csv_delimiter: str | None = None      # None = автоопределение
    csv_encoding: str = "utf-8"
    has_header: bool = True
    chunk_size: int = 10_000

    random_seed: int = 42
    n_threads: int = 4
    n_processes: int = 4

    log_level: str = "INFO"
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Загружает конфиг. Если файла нет — создаёт дефолтный и сохраняет."""

    if not path.exists():
        cfg = AppConfig()
        save_config(cfg, path)
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in AppConfig.__dataclass_fields__}
        return AppConfig(**known)
    except (json.JSONDecodeError, TypeError) as e:
        raise ConfigError(f"Не удалось прочитать конфиг {path}: {e}") from e


def save_config(cfg: AppConfig, path: Path = CONFIG_PATH) -> None:
    """Сохраняет конфиг в JSON."""
    
    try:
        path.write_text(
            json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        raise ConfigError(f"Не удалось сохранить конфиг {path}: {e}") from e