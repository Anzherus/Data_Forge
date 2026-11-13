"""Очистка данных. 
Исходные файлы не изменяются — только DataFrame в памяти."""

import pandas as pd

from utils.logger import get_logger

log = get_logger("cleaner")


def apply_cleaning(df: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, dict]:
    """Применяет правила очистки по порядку. Возвращает (df, report).

    Поддерживаемые ключи rules:
        drop_duplicates: bool
        drop_nulls: list[str] | "all"        – удалить строки с NaN в колонках
        fill_nulls: {col: value}
        fill_mean: list[str]                 – заполнить средним
        fill_median: list[str]               – заполнить медианой
        strip_strings: bool                  – убрать пробелы по краям
        lower: list[str] | "all"             – к нижнему регистру
        upper: list[str] | "all"             – к верхнему регистру
        cast_types: {col: dtype}
        drop_where: list[dict]               – [{"col": c, "op": ">", "value": v}]
    """
    report: dict = {
        "input_rows": int(len(df)),
        "operations": [],
        "removed_rows": 0,
        "fixed_cells": 0,
    }
    out = df.copy()

    # 1. дубликаты
    if rules.get("drop_duplicates"):
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        removed = before - len(out)
        report["removed_rows"] += removed
        report["operations"].append({"op": "drop_duplicates", "removed": removed})
        log.info("Удалено дубликатов: %d", removed)

    # 2. удаление строк с NaN
    drop_cols = rules.get("drop_nulls")
    if drop_cols:
        cols = list(out.columns) if drop_cols == "all" else drop_cols
        before = len(out)
        out = out.dropna(subset=cols).reset_index(drop=True)
        removed = before - len(out)
        report["removed_rows"] += removed
        report["operations"].append({"op": "drop_nulls", "cols": cols, "removed": removed})

    # 3. заполнение фиксированным значением
    for col, value in rules.get("fill_nulls", {}).items():
        if col in out.columns:
            n = int(out[col].isna().sum())
            out[col] = out[col].fillna(value)
            report["fixed_cells"] += n
            report["operations"].append({"op": "fill_nulls", "col": col, "filled": n})

    # 4. заполнение средним
    for col in rules.get("fill_mean", []):
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            n = int(out[col].isna().sum())
            mean = out[col].mean()
            out[col] = out[col].fillna(mean)
            report["fixed_cells"] += n
            report["operations"].append(
                {"op": "fill_mean", "col": col, "value": float(mean), "filled": n}
            )

    # 5. заполнение медианой
    for col in rules.get("fill_median", []):
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            n = int(out[col].isna().sum())
            med = out[col].median()
            out[col] = out[col].fillna(med)
            report["fixed_cells"] += n
            report["operations"].append(
                {"op": "fill_median", "col": col, "value": float(med), "filled": n}
            )

    # 6. strip строк
    if rules.get("strip_strings"):
        for col in out.select_dtypes(include=["object", "string"]).columns:
            out[col] = out[col].str.strip()
        report["operations"].append({"op": "strip_strings"})

    # 7. регистр
    lower = rules.get("lower")
    if lower:
        cols = list(out.select_dtypes(include=["object", "string"]).columns) if lower == "all" else lower
        for col in cols:
            if col in out.columns:
                out[col] = out[col].str.lower()
        report["operations"].append({"op": "lower", "cols": cols})

    upper = rules.get("upper")
    if upper:
        cols = list(out.select_dtypes(include=["object", "string"]).columns) if upper == "all" else upper
        for col in cols:
            if col in out.columns:
                out[col] = out[col].str.upper()
        report["operations"].append({"op": "upper", "cols": cols})

    # 8. приведение типов
    for col, dtype in rules.get("cast_types", {}).items():
        if col not in out.columns:
            continue
        try:
            if dtype in ("float", "int", float, int):
                converted = pd.to_numeric(out[col], errors="coerce")
                if dtype in ("int", int):
                    converted = converted.astype("Int64")                # nullable int
                out[col] = converted
            elif dtype in ("datetime", "datetime64[ns]", "datetime64"):
                out[col] = pd.to_datetime(out[col], errors="coerce")
            else:
                out[col] = out[col].astype(dtype)
            report["operations"].append({"op": "cast", "col": col, "dtype": str(dtype)})
        except (ValueError, TypeError) as e:
            log.warning("Не удалось привести %s к %s: %s", col, dtype, e)

    # 9. удаление по условиям
    for cond in rules.get("drop_where", []):
        col, op, value = cond["col"], cond["op"], cond["value"]
        if col not in out.columns:
            continue
        before = len(out)
        if op == ">":
            out = out[~(out[col] > value)]
        elif op == "<":
            out = out[~(out[col] < value)]
        elif op == "==":
            out = out[~(out[col] == value)]
        elif op == "!=":
            out = out[~(out[col] != value)]
        removed = before - len(out)
        report["removed_rows"] += removed
        report["operations"].append(
            {"op": "drop_where", "col": col, "op_sym": op, "value": value, "removed": removed}
        )

    out = out.reset_index(drop=True)
    report["output_rows"] = int(len(out))
    log.info(
        "Очистка завершена: %d → %d строк, исправлено ячеек: %d",
        report["input_rows"], report["output_rows"], report["fixed_cells"],
    )
    return out, report