"""Тесты трансформаций: rename, drop, computed, merge/split, datetime, sort, filter, groupby, join."""

import pandas as pd
import pytest

from modules import transformer as T
from utils.exceptions import TransformError


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "value": [10, 20, 30],
        "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
    })


def test_rename(df) -> None:
    out = T.rename_columns(df, {"name": "title"})
    assert "title" in out.columns
    assert "name" not in out.columns


def test_drop(df) -> None:
    out = T.drop_columns(df, ["value"])
    assert "value" not in out.columns


def test_add_computed(df) -> None:
    out = T.add_computed(df, "double", "value * 2")
    assert out["double"].tolist() == [20, 40, 60]


def test_merge_split(df) -> None:
    out = T.merge_columns(df, ["name", "id"], "key", sep="-")
    assert out["key"].tolist() == ["a-1", "b-2", "c-3"]
    out2 = T.split_column(out, "key", "-", ["n", "i"])
    assert out2["n"].tolist() == ["a", "b", "c"]


def test_datetime_sort_filter(df) -> None:
    out = T.transform_datetime(df, "date")
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    s = T.sort_df(df, ["value"], ascending=False)
    assert s["value"].tolist() == [30, 20, 10]
    f = T.filter_df(df, "value > 15")
    assert len(f) == 2


def test_group_agg(df) -> None:
    df2 = df.copy()
    df2["cat"] = ["x", "x", "y"]
    out = T.group_agg(df2, ["cat"], {"value": "sum"})
    assert out.loc[out["cat"] == "x", "value"].iloc[0] == 30


def test_join(df) -> None:
    right = pd.DataFrame({"id": [1, 2, 99], "extra": ["p", "q", "r"]})
    merged, report = T.join_dfs(df, right, on=["id"], how="left")
    assert len(merged) == 3
    assert report["unmatched_right"] == 1
    assert report["unmatched_left"] == 1


def test_missing_column_raises(df) -> None:
    with pytest.raises(TransformError):
        T.drop_columns(df, ["nope"])