"""Сценарии обработки: Step, Scenario, сохранение в SQLite, воспроизведение."""
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd

from modules.storage import Storage
from modules import transformer as T
from utils.exceptions import PipelineError
from utils.logger import get_logger

log = get_logger("pipeline")


@dataclass
class Step:
    op: str
    params: dict


_OPERATIONS = {
    "rename_columns": T.rename_columns,
    "drop_columns": T.drop_columns,
    "add_computed": T.add_computed,
    "merge_columns": T.merge_columns,
    "split_column": T.split_column,
    "transform_datetime": T.transform_datetime,
    "sort_df": T.sort_df,
    "filter_df": T.filter_df,
    "group_agg": T.group_agg,
}


def _apply_step(df: pd.DataFrame, step: Step) -> pd.DataFrame:
    op = step.op
    if op not in _OPERATIONS:
        raise PipelineError(f"Неизвестная операция: {op}")
    fn = _OPERATIONS[op]
    try:
        return fn(df, **step.params)
    except TypeError as e:
        raise PipelineError(f"Неверные параметры для {op}: {e}") from e


def run_pipeline(
    df: pd.DataFrame,
    steps: list[Step],
    storage: Storage | None = None,
    scenario_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Пошаговое выполнение. При падении — PipelineError с номером шага и причиной."""
    started = datetime.now()
    log.info("Запуск сценария '%s' (%d шагов)", scenario_name or "inline", len(steps))
    out = df.copy()
    history = {"input_rows": int(len(out)), "steps_run": [], "errors": []}

    for i, step in enumerate(steps, start=1):
        before_cols = set(out.columns)
        before_rows = len(out)
        try:
            out = _apply_step(out, step)
        except PipelineError:
            raise
        except Exception as e:
            msg = f"Шаг {i} ({step.op}) упал: {e}"
            log.error(msg)
            history["errors"].append(msg)
            raise PipelineError(msg) from e

        added = set(out.columns) - before_cols
        history["steps_run"].append({
            "index": i,
            "op": step.op,
            "rows_before": before_rows,
            "rows_after": int(len(out)),
            "columns_added": sorted(added),
        })

    history["output_rows"] = int(len(out))
    history["duration_sec"] = (datetime.now() - started).total_seconds()

    if storage is not None:
        storage.add_history(
            started_at=started.isoformat(timespec="seconds"),
            finished_at=datetime.now().isoformat(timespec="seconds"),
            scenario_name=scenario_name or "",
            steps_json=str([asdict(s) for s in steps]),
            input_rows=history["input_rows"],
            output_rows=history["output_rows"],
            status="ok" if not history["errors"] else "error",
            duration_sec=history["duration_sec"],
            errors="; ".join(history["errors"]),
        )

    log.info("Сценарий завершён: %d → %d строк за %.2fs",
             history["input_rows"], history["output_rows"], history["duration_sec"])
    return out, history


def save_scenario(storage: Storage, name: str, steps: list[Step]) -> int:
    data = [asdict(s) for s in steps]
    return storage.save_scenario(name, data)


def load_scenario(storage: Storage, name: str) -> list[Step]:
    raw = storage.load_scenario(name)
    if raw is None:
        raise PipelineError(f"Сценарий '{name}' не найден")
    return [Step(op=r["op"], params=r.get("params", {})) for r in raw]