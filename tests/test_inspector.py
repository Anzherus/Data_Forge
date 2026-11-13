"""Тесты инспекции: rows/cols, nulls, unique, min/max, head/tail, summary."""

import pandas as pd

from modules.inspector import head_tail, inspect, summary


def test_inspect_basic() -> None:
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "a"],
        "value": [10.0, None, 30.0],
    })
    info = inspect(df)
    assert info["rows"] == 3
    assert info["cols"] == 3
    assert info["nulls"]["value"] == 1
    assert info["unique"]["name"] == 2
    assert info["numeric_min"]["value"] == 10.0
    assert info["numeric_max"]["value"] == 30.0


def test_head_tail() -> None:
    df = pd.DataFrame({"x": range(10)})
    ht = head_tail(df, n=2)
    assert len(ht["head"]) == 2
    assert len(ht["tail"]) == 2


def test_summary() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    s = summary(df)
    assert isinstance(s, pd.DataFrame)