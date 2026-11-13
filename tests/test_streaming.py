"""Тесты потоковой обработки, NumPy vs loops, процессов vs последовательной обработки."""
from pathlib import Path

import pandas as pd
import pytest

from modules.benchmark import (
    _inmemory_stats,
    _streaming_stats,
    bench_numpy_vs_loops,
    bench_processes_vs_seq,
    bench_streaming_vs_memory,
    zscore_numpy,
    zscore_python,
)


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "stream.csv"
    rows = ["id,value"]
    for i in range(200):
        rows.append(f"{i},{i * 1.5}")
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


def test_streaming_matches_memory(csv_file: Path) -> None:
    """Потоковая обработка должна давать те же результаты, что in-memory"""
    r = bench_streaming_vs_memory(str(csv_file), chunk_size=25)
    assert r["matched"] is True
    assert r["rows"] == 200


def test_zscore_python_vs_numpy_same_result() -> None:
    import numpy as np
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    py = zscore_python(arr.tolist())
    npz = zscore_numpy(arr)
    for a, b in zip(py, npz):
        assert abs(a - b) < 1e-9


def test_bench_numpy_runs() -> None:
    r = bench_numpy_vs_loops(n=5000)
    assert r["n"] == 5000
    assert "speedup" in r
    assert "why_vectorized" in r


def test_bench_processes_matches_seq() -> None:
    """При объединении результатов не должно быть потерь/дублей"""
    df = pd.DataFrame({"x": list(range(100)) + [10000]})
    r = bench_processes_vs_seq(df, "x", n_processes=2, n_parts=2)
    assert r["matched"] is True