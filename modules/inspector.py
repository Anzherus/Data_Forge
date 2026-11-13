"""Исследование структуры набора данных."""
import pandas as pd

from utils.logger import get_logger

log = get_logger("inspector")


def inspect(df: pd.DataFrame) -> dict:
    """rows, cols, columns, dtypes, unique, nulls, min/max, min/max дат."""

    if df.empty:
        return {"rows": 0, "cols": 0, "columns": [], "dtypes": {}}

    result: dict = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "unique": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
        "nulls_total": int(df.isna().sum().sum()),
    }

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        result["numeric_min"] = numeric.min().to_dict()
        result["numeric_max"] = numeric.max().to_dict()

    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"])
    if not datetime_cols.empty:
        result["date_min"] = {
            c: str(v) for c, v in datetime_cols.min().items()
        }
        result["date_max"] = {
            c: str(v) for c, v in datetime_cols.max().items()
        }

    log.info("Инспекция: %d строк, %d колонок", result["rows"], result["cols"])
    return result


def head_tail(df: pd.DataFrame, n: int = 5) -> dict:
    """Первые и последние n записей."""

    return {"head": df.head(n), "tail": df.tail(n)}


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Общая сводка по набору (как df.describe(include='all'))."""
    
    return df.describe(include="all").transpose()