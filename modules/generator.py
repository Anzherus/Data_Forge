"""Генерация тестовых данных: корректные + специально испорченные, с seed."""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("generator")


def generate_dataset(
    rows: int,
    cols: int,
    categories: int = 5,
    numeric_range: tuple[float, float] = (0, 1000),
    date_range: tuple[str, str] = ("2025-01-01", "2025-12-31"),
    null_pct: float = 0.0,
    dup_pct: float = 0.0,
    invalid_pct: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Генерирует набор с заданными параметрами загрязнения.

    null_pct / dup_pct / invalid_pct – доли от 0.0 до 1.0.
    Повторный вызов с тем же seed даёт идентичный результат."""
    
    random.seed(seed)
    np.random.seed(seed)

    start = datetime.fromisoformat(date_range[0])
    end = datetime.fromisoformat(date_range[1])
    days = max((end - start).days, 1)

    n_num = max(cols - 3, 1)
    data: dict = {
        "record_id": np.arange(1, rows + 1),
        "date": [
            (start + timedelta(days=random.randint(0, days))).date().isoformat()
            for _ in range(rows)
        ],
        "category": [f"cat_{random.randint(1, categories)}" for _ in range(rows)],
    }
    for i in range(n_num):
        data[f"value_{i}"] = np.random.uniform(*numeric_range, size=rows)

    df = pd.DataFrame(data)

    # порча: пропуски
    if null_pct > 0:
        n_null = int(rows * null_pct)
        for col in df.columns:
            if col == "record_id":
                continue
            idx = np.random.choice(df.index, size=min(n_null, len(df)), replace=False)
            df.loc[idx, col] = None

    # порча: некорректные значения 
    if invalid_pct > 0:
        n_bad = int(rows * invalid_pct)
        if "value_0" in df.columns and n_bad > 0:
            # заранее приводим колонку к object, иначе pandas 3.x не даст записать строку
            df["value_0"] = df["value_0"].astype(object)
            idx = np.random.choice(df.index, size=min(n_bad, len(df)), replace=False)
            df.loc[idx, "value_0"] = "not_a_number"
        if n_bad > 0:
            df["date"] = df["date"].astype(object)
            idx = np.random.choice(df.index, size=min(n_bad, len(df)), replace=False)
            df.loc[idx, "date"] = "bad-date"
            
    # порча: дубликаты 
    if dup_pct > 0:
        n_dup = int(rows * dup_pct)
        if n_dup > 0:
            dup_rows = df.sample(n=min(n_dup, len(df)), random_state=seed)
            df = pd.concat([df, dup_rows], ignore_index=True)

    log.info("Сгенерировано %d строк, %d колонок (seed=%d)", len(df), df.shape[1], seed)
    return df