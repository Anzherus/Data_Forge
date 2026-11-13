"""Тесты аномалий: IQR, z-score, bounds, отсутствие автоудаления."""

import pandas as pd
import pytest

from modules.anomaly_detector import detect_bounds, detect_iqr, detect_zscore
from utils.exceptions import ValidationError


def test_iqr_finds_outlier() -> None:
    """IQR находит выброс, criterion и record сохраняются"""
    
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
    found = detect_iqr(df, "x")
    assert len(found) == 1
    assert found[0]["field"] == "x"
    assert found[0]["value"] == 1000
    assert "IQR" in found[0]["criterion"]
    assert "record" in found[0]


def test_zscore_finds_outlier() -> None:
    df = pd.DataFrame({"x": [10, 10, 10, 10, 10, 10, 1000]})
    found = detect_zscore(df, "x", threshold=2.0)
    assert len(found) >= 1
    assert any(a["value"] == 1000 for a in found)


def test_bounds() -> None:
    df = pd.DataFrame({"x": [-5, 1, 5, 100]})
    found = detect_bounds(df, "x", low=0, high=10)
    assert len(found) == 2


def test_no_autodelete() -> None:
    """Аномальные записи не удаляются автоматически"""

    df = pd.DataFrame({"x": [1, 2, 3, 1000]})
    before = len(df)
    detect_iqr(df, "x")
    assert len(df) == before  # исходный не изменён


def test_bad_column() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValidationError):
        detect_iqr(df, "nope")