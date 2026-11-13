"""Тесты загрузки: CSV, JSON, отсутствующие файлы, неподдерживаемые форматы, потоковое чтение."""
import json
from pathlib import Path

import pandas as pd
import pytest

from modules.loader import ChunkIterator, iter_chunks, load_file, load_multiple
from utils.exceptions import LoadError, UnsupportedFormatError


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    p.write_text("id,name,value\n1,Alice,10\n2,Bob,20\n", encoding="utf-8")
    return p


@pytest.fixture
def json_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.json"
    p.write_text(
        json.dumps([
            {"id": 1, "name": "Alice", "value": 10},
            {"id": 2, "name": "Bob", "value": 20},
        ]),
        encoding="utf-8",
    )
    return p


def test_load_csv(csv_file: Path) -> None:
    df = load_file(csv_file)
    assert len(df) == 2
    assert list(df.columns) == ["id", "name", "value"]


def test_load_json(json_file: Path) -> None:
    df = load_file(json_file)
    assert len(df) == 2
    assert "name" in df.columns


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LoadError):
        load_file(tmp_path / "nope.csv")


def test_unsupported_format(tmp_path: Path) -> None:
    p = tmp_path / "file.abc"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        load_file(p)


def test_load_multiple(csv_file: Path, json_file: Path) -> None:
    df = load_multiple([csv_file, json_file])
    assert len(df) == 4


def test_iter_chunks(csv_file: Path) -> None:
    chunks = list(iter_chunks(csv_file, chunk_size=1))
    assert len(chunks) == 2
    assert all(isinstance(c, pd.DataFrame) for c in chunks)


def test_chunk_iterator(csv_file: Path) -> None:
    it = ChunkIterator(csv_file, chunk_size=1)
    count = 0
    for chunk in it:
        assert isinstance(chunk, pd.DataFrame)
        count += 1
    assert count == 2
    assert it.chunks_read == 2