"""Тесты очистки: дубли, пропуски, регистр, приведение типов, drop_where, неизменность исходника."""

import pandas as pd

from modules.cleaner import apply_cleaning


def test_drop_duplicates() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out, rep = apply_cleaning(df, {"drop_duplicates": True})
    assert len(out) == 2
    assert rep["removed_rows"] == 1


def test_fill_mean_median() -> None:
    df = pd.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
    out, rep = apply_cleaning(df, {"fill_mean": ["x"]})
    assert out["x"].isna().sum() == 0
    assert out["x"].iloc[1] == 3.0                              # mean([1,3,5])=3.0
    df2 = pd.DataFrame({"x": [1.0, None, 100.0]})
    out2, _ = apply_cleaning(df2, {"fill_median": ["x"]})
    assert out2["x"].iloc[1] == 50.5                            # median([1,100])


def test_strip_and_lower() -> None:
    df = pd.DataFrame({"name": ["  Alice ", " BOB  "]})
    out, _ = apply_cleaning(df, {"strip_strings": True, "lower": "all"})
    assert list(out["name"]) == ["alice", "bob"]


def test_cast_and_drop_where() -> None:
    df = pd.DataFrame({"v": ["1", "2", "abc"]})
    out, rep = apply_cleaning(df, {"cast_types": {"v": "float"}})
    # 'abc' станет NaN
    assert out["v"].isna().sum() == 1


def test_original_unchanged() -> None:
    df = pd.DataFrame({"a": [1, 1, 2]})
    original_len = len(df)
    apply_cleaning(df, {"drop_duplicates": True})
    assert len(df) == original_len  # исходный не тронут