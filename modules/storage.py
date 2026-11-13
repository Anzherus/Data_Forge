"""Хранилище: настройки, сценарии, история запусков, отчёты (SQLite)."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.exceptions import StorageError
from utils.logger import get_logger

log = get_logger("storage")
#Cхема БД:
#    settings – настройки приложения(key-value)
#    scenarios – сохранение сценарии обработки
#    history – история запусков сценариев
#    reports – отчёты
_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scenarios (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    steps_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    source_files  TEXT,
    scenario_name TEXT,
    steps_json    TEXT,
    input_rows    INTEGER,
    output_rows   INTEGER,
    fixed_rows    INTEGER,
    removed_rows  INTEGER,
    errors        TEXT,
    duration_sec  REAL,
    status        TEXT
);
CREATE TABLE IF NOT EXISTS reports (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Storage:
    """Обёртка над SQLite. Все чтения/записи – через методы."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            log.info("Storage инициализирован: %s", self.db_path)
        except sqlite3.Error as e:
            raise StorageError(f"Ошибка инициализации БД: {e}") from e

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # settings
    def set_setting(self, key: str, value: Any) -> None:
        self._require_conn()
        try:
            self._conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"set_setting failed: {e}") from e

    def get_setting(self, key: str, default: Any = None) -> Any:
        self._require_conn()
        row = self._conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row["value"]) if row else default

    # scenarios
    def save_scenario(self, name: str, steps: list[dict]) -> int:
        self._require_conn()
        now = datetime.now().isoformat(timespec="seconds")
        try:
            cur = self._conn.execute(
                "INSERT INTO scenarios(name, steps_json, created_at, updated_at) "
                "VALUES(?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "steps_json=excluded.steps_json, updated_at=excluded.updated_at",
                (name, json.dumps(steps, ensure_ascii=False), now, now),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except sqlite3.Error as e:
            raise StorageError(f"save_scenario failed: {e}") from e

    def load_scenario(self, name: str) -> list[dict] | None:
        self._require_conn()
        row = self._conn.execute(
            "SELECT steps_json FROM scenarios WHERE name=?", (name,)
        ).fetchone()
        return json.loads(row["steps_json"]) if row else None

    def list_scenarios(self) -> list[str]:
        self._require_conn()
        rows = self._conn.execute("SELECT name FROM scenarios ORDER BY name").fetchall()
        return [r["name"] for r in rows]

    # history 
    def add_history(self, **fields: Any) -> int:
        self._require_conn()
        cols = ", ".join(fields.keys())
        ph = ", ".join("?" * len(fields))
        try:
            cur = self._conn.execute(
                f"INSERT INTO history({cols}) VALUES({ph})",
                tuple(fields.values()),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except sqlite3.Error as e:
            raise StorageError(f"add_history failed: {e}") from e

    def list_history(self, limit: int = 50) -> list[dict]:
        self._require_conn()
        rows = self._conn.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # reports
    def save_report(self, kind: str, payload: dict) -> int:
        self._require_conn()
        try:
            cur = self._conn.execute(
                "INSERT INTO reports(kind, payload, created_at) VALUES(?, ?, ?)",
                (kind, json.dumps(payload, ensure_ascii=False, default=str),
                 datetime.now().isoformat(timespec="seconds")),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except sqlite3.Error as e:
            raise StorageError(f"save_report failed: {e}") from e

    # internal 
    def _require_conn(self) -> None:
        if self._conn is None:
            raise StorageError("Соединение с БД закрыто")