"""Экспорт результатов в CSV/JSON/XLSX + сохранение отчётов."""

import json
from pathlib import Path

import pandas as pd

from utils.exceptions import ExportError
from utils.logger import get_logger

log = get_logger("exporter")


def _infer_format(path: Path, fmt: str | None) -> str:
    if fmt:
        return fmt.lower()
    ext = path.suffix.lower().lstrip(".")
    if ext in {"csv", "json", "xlsx"}:
        return ext
    raise ExportError(f"Не удалось определить формат для {path}")


def export(
    df: pd.DataFrame,
    path: str | Path,
    *,
    columns: list[str] | None = None,
    fmt: str | None = None,
) -> None:
    """Экспорт DataFrame с выбором колонок. Исходный df не изменяется."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = _infer_format(p, fmt)

    out = df.copy()
    if columns:
        missing = [c for c in columns if c not in out.columns]
        if missing:
            raise ExportError(f"Колонки для экспорта не найдены: {missing}")
        out = out[columns]

    try:
        if fmt == "csv":
            out.to_csv(p, index=False, encoding="utf-8")
        elif fmt == "json":
            out.to_json(p, orient="records", force_ascii=False, indent=2)
        elif fmt == "xlsx":
            out.to_excel(p, index=False, engine="openpyxl")
        else:
            raise ExportError(f"Формат {fmt} не поддерживается")
    except (OSError, ValueError) as e:
        raise ExportError(f"Ошибка экспорта в {p}: {e}") from e

    log.info("Экспортировано %d строк в %s (%s)", len(out), p, fmt)


def save_report(report: dict, path: str | Path, fmt: str = "json") -> None:
    """Сохраняет отчёт о выполненных операциях (JSON или CSV)."""
    
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if fmt == "json":
            p.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        elif fmt == "csv":
            # разворачиваем плоские части отчёта в таблицу
            rows = []
            for op in report.get("operations", []):
                rows.append(op)
            pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8")
        else:
            raise ExportError(f"Формат отчёта {fmt} не поддерживается")
    except OSError as e:
        raise ExportError(f"Не удалось сохранить отчёт {p}: {e}") from e
    log.info("Отчёт сохранён: %s", p)