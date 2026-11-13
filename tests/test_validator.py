"""Тесты валидации: nulls, duplicates, constraints, outliers, bad_dates."""
import pandas as pd

from modules.validator import ValidationRules, validate


def test_nulls_and_duplicates() -> None:
    df = pd.DataFrame({
        "id": [1, 2, 2, 4],
        "value": [10.0, None, None, 40.0],
    })
    result = validate(df)
    assert result["nulls"]["count"] == 2
    assert result["duplicates"]["count"] == 1
    assert "value" in result["nulls"]["fields"]


def test_constraints() -> None:
    df = pd.DataFrame({"age": [10, 150, 30]})
    rules = ValidationRules(constraints={"age": {"min": 0, "max": 120}})
    result = validate(df, rules)
    assert result["constraints"]["count"] == 1
    assert "age" in result["constraints"]["fields"]


def test_outliers() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
    result = validate(df)
    assert result["outliers"]["count"] >= 1
    assert "x" in result["outliers"]["fields"]


def test_bad_dates() -> None:
    df = pd.DataFrame({
        "dt": ["2025-01-01", "not-a-date", "2025-03-01", "2025-04-01"],
    })
    result = validate(df)
    assert result["bad_dates"]["count"] >= 1