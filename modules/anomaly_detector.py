"""Поиск аномалий: IQR, z-score, пользовательские границы.

Аномальные записи НЕ удаляются — только возвращаются как список.
"""
import numpy as np
import pandas as pd

from utils.exceptions import ValidationError
from utils.logger import get_logger

log = get_logger("anomaly_detector")


def _row_to_dict(row: pd.Series) -> dict:
    return {str(k): (None if pd.isna(v) else v.item() if hasattr(v, "item") else v)
            for k, v in row.items()}


def detect_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> list[dict]:
    """Выбросы по межквартильному размаху: < Q1 - k*IQR или > Q3 + k*IQR."""
    if column not in df.columns:
        raise ValidationError(f"Колонка не найдена: {column}")
    s = pd.to_numeric(df[column], errors="coerce")
    valid = s.dropna()
    if len(valid) < 4:
        return []
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return []
    low, high = q1 - factor * iqr, q3 + factor * iqr
    mask = (s < low) | (s > high)
    result = []
    for idx in df.index[mask.fillna(False)]:
        result.append({
            "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "record": _row_to_dict(df.loc[idx]),
            "field": column,
            "value": float(s.loc[idx]),
            "criterion": f"IQR(factor={factor}): [{low:.4f}, {high:.4f}]",
        })
    log.info("detect_iqr(%s): %d аномалий", column, len(result))
    return result


def detect_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> list[dict]:
    """Выбросы по отклонению от среднего: |x - mean| / std > threshold."""
    if column not in df.columns:
        raise ValidationError(f"Колонка не найдена: {column}")
    s = pd.to_numeric(df[column], errors="coerce")
    valid = s.dropna()
    if len(valid) < 2:
        return []
    mu, sigma = valid.mean(), valid.std()
    if sigma == 0 or pd.isna(sigma):
        return []
    z = (s - mu) / sigma
    mask = z.abs() > threshold
    result = []
    for idx in df.index[mask.fillna(False)]:
        result.append({
            "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "record": _row_to_dict(df.loc[idx]),
            "field": column,
            "value": float(s.loc[idx]),
            "criterion": f"Z-score(threshold={threshold}): z={float(z.loc[idx]):.3f}",
        })
    log.info("detect_zscore(%s): %d аномалий", column, len(result))
    return result


def detect_bounds(
    df: pd.DataFrame, column: str, low: float, high: float
) -> list[dict]:
    """Значения вне пользовательских границ [low, high]."""
    if column not in df.columns:
        raise ValidationError(f"Колонка не найдена: {column}")
    s = pd.to_numeric(df[column], errors="coerce")
    mask = (s < low) | (s > high)
    result = []
    for idx in df.index[mask.fillna(False)]:
        result.append({
            "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "record": _row_to_dict(df.loc[idx]),
            "field": column,
            "value": float(s.loc[idx]),
            "criterion": f"Bounds: [{low}, {high}]",
        })
    log.info("detect_bounds(%s): %d аномалий", column, len(result))
    return result