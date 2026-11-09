import pandas as pd
import pytest

from modules.analyzer import (
    categorical_stats,
    compare_groups,
    correlations,
    numeric_stats,
    pivot_table,
    temporal_stats,
)
from utils.exceptions import ValidationError


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10],
        "cat": ["a", "a", "b", "b", "b"],
        "dt": pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-02-01", "2025-02-02", "2025-03-01",
        ]),
    })


def test_numeric_stats(df) -> None:
    s = numeric_stats(df, ["x"])
    assert s["x"]["count"] == 5
    assert s["x"]["mean"] == 3.0
    assert s["x"]["median"] == 3.0
    assert s["x"]["min"] == 1
    assert s["x"]["max"] == 5
    assert s["x"]["q1"] == 2.0
    assert s["x"]["q3"] == 4.0


def test_categorical_stats(df) -> None:
    c = categorical_stats(df, ["cat"])
    assert c["cat"]["unique"] == 2
    assert c["cat"]["top"][0] == "b"
    assert c["cat"]["freq"]["b"] == 3


def test_temporal_stats(df) -> None:
    t = temporal_stats(df, "dt", freq="M")
    assert len(t) == 3
    assert t["count"].sum() == 5


def test_correlations(df) -> None:
    c = correlations(df, ["x", "y"])
    assert "note" in c
    assert "причинно" in c["note"].lower() or "НЕ" in c["note"]
    assert any(p["a"] == "x" and p["b"] == "y" for p in c["high_pairs"])


def test_pivot(df) -> None:
    p = pivot_table(df, index="cat", columns="dt", values="x", aggfunc="mean")
    assert isinstance(p, pd.DataFrame)


def test_compare_groups(df) -> None:
    r = compare_groups(df, "cat", "x")
    assert "groups" in r
    assert "note" in r


def test_temporal_bad_column(df) -> None:
    with pytest.raises(ValidationError):
        temporal_stats(df, "nope")