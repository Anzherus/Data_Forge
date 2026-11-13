"""Тесты экспорта: CSV, JSON, XLSX, выбор колонок, отчёты."""
import json
from pathlib import Path

import pandas as pd
import pytest

from modules.exporter import export, save_report
from utils.exceptions import ExportError


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})


def test_export_csv(df, tmp_path: Path) -> None:
    p = tmp_path / "out.csv"
    export(df, p)
    loaded = pd.read_csv(p)
    assert len(loaded) == 2
    assert list(loaded.columns) == ["a", "b"]


def test_export_json(df, tmp_path: Path) -> None:
    p = tmp_path / "out.json"
    export(df, p, fmt="json")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["a"] == 1


def test_export_xlsx(df, tmp_path: Path) -> None:
    p = tmp_path / "out.xlsx"
    export(df, p, fmt="xlsx")
    loaded = pd.read_excel(p, engine="openpyxl")
    assert len(loaded) == 2


def test_export_columns_subset(df, tmp_path: Path) -> None:
    p = tmp_path / "sub.csv"
    export(df, p, columns=["b"])
    loaded = pd.read_csv(p)
    assert list(loaded.columns) == ["b"]


def test_export_missing_column_raises(df, tmp_path: Path) -> None:
    with pytest.raises(ExportError):
        export(df, tmp_path / "x.csv", columns=["nope"])


def test_save_report_json(tmp_path: Path) -> None:
    report = {"input_rows": 10, "output_rows": 8, "operations": [{"op": "drop"}]}
    p = tmp_path / "rep.json"
    save_report(report, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["input_rows"] == 10