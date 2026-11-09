"""Преобразования данных: 9 операций + join с обработкой unmatched."""
import pandas as pd

from utils.exceptions import TransformError
from utils.logger import get_logger

log = get_logger("transformer")


def rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    missing = [c for c in mapping if c not in df.columns]
    if missing:
        raise TransformError(f"Колонки для переименования не найдены: {missing}")
    return df.rename(columns=mapping)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise TransformError(f"Колонки для удаления не найдены: {missing}")
    return df.drop(columns=columns)


def add_computed(df: pd.DataFrame, name: str, expression: str) -> pd.DataFrame:
    """Вычисляемый столбец. expression — pandas-выражение через df.eval."""
    try:
        out = df.copy()
        out[name] = out.eval(expression)
        return out
    except Exception as e:
        raise TransformError(f"Ошибка вычисления '{name} = {expression}': {e}") from e


def merge_columns(
    df: pd.DataFrame, columns: list[str], new_name: str, sep: str = " "
) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise TransformError(f"Колонки для объединения не найдены: {missing}")
    out = df.copy()
    out[new_name] = out[columns].astype(str).agg(sep.join, axis=1)
    return out


def split_column(
    df: pd.DataFrame, column: str, sep: str, new_names: list[str]
) -> pd.DataFrame:
    if column not in df.columns:
        raise TransformError(f"Колонка для разделения не найдена: {column}")
    out = df.copy()
    parts = out[column].astype(str).str.split(sep, n=len(new_names) - 1, expand=True)
    for i, name in enumerate(new_names):
        out[name] = parts[i] if i in parts.columns else None
    return out


def transform_datetime(
    df: pd.DataFrame, column: str, fmt: str | None = None
) -> pd.DataFrame:
    if column not in df.columns:
        raise TransformError(f"Колонка для преобразования даты не найдена: {column}")
    out = df.copy()
    out[column] = pd.to_datetime(out[column], format=fmt, errors="coerce")
    return out


def sort_df(df: pd.DataFrame, by: list[str], ascending: bool = True) -> pd.DataFrame:
    missing = [c for c in by if c not in df.columns]
    if missing:
        raise TransformError(f"Колонки для сортировки не найдены: {missing}")
    return df.sort_values(by=by, ascending=ascending).reset_index(drop=True)


def filter_df(df: pd.DataFrame, expression: str) -> pd.DataFrame:
    try:
        return df.query(expression).reset_index(drop=True)
    except Exception as e:
        raise TransformError(f"Ошибка фильтрации '{expression}': {e}") from e


def group_agg(df: pd.DataFrame, by: list[str], aggs: dict[str, str]) -> pd.DataFrame:
    missing = [c for c in by if c not in df.columns]
    if missing:
        raise TransformError(f"Колонки для группировки не найдены: {missing}")
    try:
        return df.groupby(by, dropna=False).agg(aggs).reset_index()
    except Exception as e:
        raise TransformError(f"Ошибка группировки: {e}") from e


def join_dfs(
    left: pd.DataFrame, right: pd.DataFrame, on: list[str], how: str = "inner"
) -> tuple[pd.DataFrame, dict]:
    """Возвращает (df, report) с unmatched-записями."""
    for side, d in (("left", left), ("right", right)):
        missing = [c for c in on if c not in d.columns]
        if missing:
            raise TransformError(f"В {side} нет ключевых колонок: {missing}")

    left_keys = set(map(tuple, left[on].astype(str).values))
    right_keys = set(map(tuple, right[on].astype(str).values))
    unmatched_left = left_keys - right_keys
    unmatched_right = right_keys - left_keys

    merged = left.merge(right, on=on, how=how, suffixes=("_x", "_y"))
    report = {
        "how": how,
        "keys": on,
        "rows_left": int(len(left)),
        "rows_right": int(len(right)),
        "rows_result": int(len(merged)),
        "unmatched_left": int(len(unmatched_left)),
        "unmatched_right": int(len(unmatched_right)),
    }
    log.info("Join %s: %d + %d → %d (unmatched L=%d, R=%d)",
             how, len(left), len(right), len(merged),
             len(unmatched_left), len(unmatched_right))
    return merged, report