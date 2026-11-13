"""Тесты обработки ошибочных файлов: отсутствующий файл, битый JSON."""
from pathlib import Path

import pytest

from modules.loader import load_file
from utils.exceptions import CorruptedFileError, LoadError


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LoadError):
        load_file(tmp_path / "nope.csv")


def test_broken_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CorruptedFileError):
        load_file(p)