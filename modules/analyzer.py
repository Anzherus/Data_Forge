"""Аналитика: числовая, категориальная, временная, корреляции, сводные."""
import numpy as np
import pandas as pd

from utils.exceptions import ValidationError
from utils.logger import get_logger

log = get_logger("analyzer")


def numeric_stats(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """count, sum, mean, median, min, max, std, quartiles для числовых колонок."""
    cols = columns or list(df.select_dtypes(include="number").columns)
    result: dict = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            result[col] = {"count": 0}
            continue
        result[col] = {
            "count": int(s.count()),
            "sum": float(s.sum()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "q1": float(s.quantile(0.25)),
            "q2": float(s.quantile(0.50)),
            "q3": float(s.quantile(0.75)),
        }
    log.info("numeric_stats: %d колонок", len(result))
    return result


def categorical_stats(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """unique count, частоты, топ-значения для категориальных колонок."""
    cols = columns or [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    result: dict = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        freq = s.value_counts()
        result[col] = {
            "unique": int(s.nunique()),
            "total": int(s.count()),
            "freq": {str(k): int(v) for k, v in freq.items()},
            "top": [str(x) for x in freq.head(5).index.tolist()],
        }
    log.info("categorical_stats: %d колонок", len(result))
    return result



_FREQ_ALIASES = {
    "M": "ME",
    "Y": "YE",
    "A": "YE",
    "Q": "QE",
}


def _normalize_freq(freq: str) -> str:
    """Приводит старые частоты pandas к новым (M→ME, Y→YE, Q→QE)."""
    return _FREQ_ALIASES.get(freq.upper(), freq)

def temporal_stats(df: pd.DataFrame, column: str, freq: str = "D") -> pd.DataFrame:
    """Агрегация по дням (D), неделям (W), месяцам (M) и т.д."""
    if column not in df.columns:
        raise ValidationError(f"Колонка не найдена: {column}")
    s = df[column]
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce")
    tmp = pd.DataFrame({"_dt": s, "_count": 1})
    freq = _normalize_freq(freq)
    out = (
        tmp.dropna(subset=["_dt"])
        .set_index("_dt")
        .resample(freq)
        .agg(count=("_count", "sum"))
        .reset_index()
    )
    return out


def correlations(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Матрица корреляций + пары с высоким |r| + оговорка про причинность."""
    cols = columns or list(df.select_dtypes(include="number").columns)
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return {
            "matrix": {},
            "high_pairs": [],
            "note": "Для корреляции нужно минимум 2 числовых столбца.",
        }

    corr = df[cols].corr(numeric_only=True)
    high_pairs: list[dict] = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r = corr.loc[c1, c2]
            if pd.notna(r) and abs(r) >= 0.7:
                high_pairs.append({"a": c1, "b": c2, "r": float(r)})

    high_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    note = (
        "Корреляция показывает статистическую связь, но НЕ доказывает "
        "причинно-следственную зависимость. Высокий |r| может объясняться "
        "третьим фактором, трендом или случайностью на малой выборке."
    )
    log.info("correlations: %d пар с |r|>=0.7", len(high_pairs))
    return {
        "matrix": {c: corr[c].to_dict() for c in corr.columns},
        "high_pairs": high_pairs,
        "note": note,
    }


def pivot_table(
    df: pd.DataFrame, index: str, columns: str, values: str, aggfunc: str = "mean"
) -> pd.DataFrame:
    for c in (index, columns, values):
        if c not in df.columns:
            raise ValidationError(f"Колонка для сводной не найдена: {c}")
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)


def compare_groups(
    df: pd.DataFrame, group_col: str, value_col: str
) -> dict:
    """Сравнение групп: средние и разница."""
    for c in (group_col, value_col):
        if c not in df.columns:
            raise ValidationError(f"Колонка не найдена: {c}")
    grouped = df.groupby(group_col)[value_col].agg(["count", "mean", "median", "std"])
    means = grouped["mean"].dropna()
    diff = None
    if len(means) >= 2:
        diff = float(means.max() - means.min())
    return {
        "groups": {
            str(k): {k2: (float(v2) if pd.notna(v2) else None) for k2, v2 in row.items()}
            for k, row in grouped.iterrows()
        },
        "max_minus_min_mean": diff,
        "note": (
            "Разница средних не является статистическим тестом. "
            "Для выводов о значимости нужен t-test или ANOVA."
        ),
    }