"""Проверка качества данных: 7 типов проблем."""
import warnings
from dataclasses import dataclass, field

import pandas as pd

from utils.logger import get_logger

log = get_logger("validator")


@dataclass
class ValidationRules:
    """Настраиваемые правила проверки."""
    constraints: dict[str, dict] = field(default_factory=dict)
    iqr_factor: float = 1.5
    zscore_threshold: float = 3.0


def _check_nulls(df: pd.DataFrame) -> dict:
    """Считает пустые ячейки по колонкам. count -- общее число NaN во всём DataFrame"""
    counts = df.isna().sum()
    fields = {c: int(v) for c, v in counts.items() if v > 0}
    return {
        "count": int(sum(fields.values())),
        "fields": list(fields.keys()),
        "details": fields,
    }


def _check_duplicates(df: pd.DataFrame) -> dict:
    """Полные дубликаты строк (keep='first'). sample_indices--первые 20 индексов"""
    mask = df.duplicated(keep="first")
    return {
        "count": int(mask.sum()),
        "fields": list(df.columns),
        "sample_indices": df.index[mask].tolist()[:20],
    }


def _check_wrong_types(df: pd.DataFrame) -> dict:
    """Ищет колонки, где смешаны типы (object-колонки с числами/датами)."""
    problems: dict[str, int] = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].dropna()
        if s.empty:
            continue
        numeric_like = pd.to_numeric(s, errors="coerce")
        mixed = numeric_like.notna().sum()
        if 0 < mixed < len(s):
            problems[col] = int(len(s) - mixed)
    return {
        "count": int(sum(problems.values())),
        "fields": list(problems.keys()),
        "details": problems,
    }


def _check_constraints(df: pd.DataFrame, rules: ValidationRules) -> dict:
    """Проверяет колонки по рпавилам constraints: min/max/allowed."""
    problems: dict[str, int] = {}
    for col, rule in rules.constraints.items():
        if col not in df.columns:
            continue
        s = df[col]
        bad = pd.Series(False, index=df.index)
        if "min" in rule and pd.api.types.is_numeric_dtype(s):
            bad |= s < rule["min"]
        if "max" in rule and pd.api.types.is_numeric_dtype(s):
            bad |= s > rule["max"]
        if "allowed" in rule:
            bad |= ~s.isin(rule["allowed"]) & s.notna()
        if bad.any():
            problems[col] = int(bad.sum())
    return {
        "count": int(sum(problems.values())),
        "fields": list(problems.keys()),
        "details": problems,
    }


def _check_bad_dates(df: pd.DataFrame) -> dict:
    """Ищет невалидные даты в object-колонках, где >=50% значений похожи на даты"""
    problems: dict[str, int] = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().sum() >= len(s) * 0.5:
            bad = parsed.isna().sum()
            if bad > 0:
                problems[col] = int(bad)
    return {
        "count": int(sum(problems.values())),
        "fields": list(problems.keys()),
        "details": problems,
    }


def _check_outliers(df: pd.DataFrame, rules: ValidationRules) -> dict:
    """Выбросы по IQR: значения вне [Q1-k*IQR, Q3+k*IQR], k=rules.iqr_factor"""
    problems: dict[str, int] = {}
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - rules.iqr_factor * iqr, q3 + rules.iqr_factor * iqr
        bad = ((s < low) | (s > high)).sum()
        if bad > 0:
            problems[col] = int(bad)
    return {
        "count": int(sum(problems.values())),
        "fields": list(problems.keys()),
        "details": problems,
    }


def _check_suspicious(df: pd.DataFrame) -> dict:
    """Противоречия: одинаковый id с разными значениями."""
    problems: dict[str, int] = {}
    for col in df.columns:
        if "id" in col.lower() and df[col].notna().any():
            grouped = df.groupby(col).nunique(dropna=True)
            conflicting = (grouped > 1).any(axis=1).sum()
            if conflicting > 0:
                problems[col] = int(conflicting)
    return {
        "count": int(sum(problems.values())),
        "fields": list(problems.keys()),
        "details": problems,
    }


def validate(df: pd.DataFrame, rules: ValidationRules | None = None) -> dict:
    """Находит 7 типов проблем. Возвращает {problem: {count, fields, ...}}."""
    rules = rules or ValidationRules()
    log.info("Валидация: %d строк, %d колонок", len(df), df.shape[1])
    result = {
        "nulls": _check_nulls(df),
        "duplicates": _check_duplicates(df),
        "wrong_types": _check_wrong_types(df),
        "constraints": _check_constraints(df, rules),
        "bad_dates": _check_bad_dates(df),
        "outliers": _check_outliers(df, rules),
        "suspicious": _check_suspicious(df),
    }
    total = sum(r["count"] for r in result.values())
    log.info("Найдено проблем: %d", total)
    return result