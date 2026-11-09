"""Загрузка данных из CSV/JSON/TXT/XLSX, объединение, потоковое чтение."""
import csv
from pathlib import Path
from typing import Iterator

import pandas as pd

from utils.exceptions import (
    CorruptedFileError,
    EncodingError,
    LoadError,
    UnsupportedFormatError,
)
from utils.logger import get_logger

log = get_logger("loader")

_SUPPORTED = {"csv", "json", "txt", "xlsx"}


def _detect_format(path: Path) -> str:
    """Определяет формат по расширению."""
    ext = path.suffix.lower().lstrip(".")
    if ext not in _SUPPORTED:
        raise UnsupportedFormatError(f"Формат .{ext} не поддерживается: {path}")
    return ext


def _detect_delimiter(path: Path, encoding: str) -> str:
    """Автоопределение разделителя CSV через csv.Sniffer, дефолт — ';' или ','."""
    try:
        with path.open("r", encoding=encoding, newline="") as f:
            sample = f.read(8192)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except (csv.Error, UnicodeDecodeError):
        return ","


def load_file(
    path: str | Path,
    *,
    fmt: str | None = None,
    delimiter: str | None = None,
    encoding: str = "utf-8",
    has_header: bool = True,
) -> pd.DataFrame:
    """Загружает один файл. fmt = 'csv'|'json'|'txt'|'xlsx' или авто по расширению."""
    p = Path(path)
    if not p.exists():
        raise LoadError(f"Файл не найден: {p}")

    fmt = (fmt or _detect_format(p)).lower()
    log.info("Загрузка файла: %s (формат=%s, кодировка=%s)", p, fmt, encoding)

    try:
        if fmt in {"csv", "txt"}:
            sep = delimiter or _detect_delimiter(p, encoding)
            if fmt == "txt" and delimiter is None:
                sep = "\t"
            df = pd.read_csv(
                p,
                sep=sep,
                encoding=encoding,
                header=0 if has_header else None,
            )
        elif fmt == "json":
            df = pd.read_json(p, encoding=encoding)
            # нормализация вложенных объектов, если есть
            if df.shape[1] == 1 and df.iloc[:, 0].dtype == object:
                df = pd.json_normalize(df.iloc[:, 0])
        elif fmt == "xlsx":
            df = pd.read_excel(p, engine="openpyxl")
        else:
            raise UnsupportedFormatError(f"Формат {fmt} не поддерживается")
    except UnicodeDecodeError as e:
        raise EncodingError(f"Ошибка кодировки при чтении {p}: {e}") from e
    except pd.errors.ParserError as e:
        raise CorruptedFileError(f"Повреждённый файл {p}: {e}") from e
    except ValueError as e:
        raise CorruptedFileError(f"Не удалось прочитать {p}: {e}") from e

    log.info("Загружено: %s — %d строк, %d колонок", p, len(df), df.shape[1])
    return df


def load_multiple(
    files: list[str | Path],
    *,
    schema_mapping: dict[str, dict[str, str]] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Загружает несколько файлов и объединяет совместимые наборы.

    schema_mapping: {filename: {old_col: new_col}} для приведения колонок.
    """
    frames: list[pd.DataFrame] = []
    loaded, rejected = 0, 0

    for f in files:
        try:
            df = load_file(f, **kwargs)
            if schema_mapping and str(f) in schema_mapping:
                df = df.rename(columns=schema_mapping[str(f)])
            frames.append(df)
            loaded += 1
        except LoadError as e:
            log.warning("Пропущен файл %s: %s", f, e)
            rejected += 1

    if not frames:
        raise LoadError("Ни один файл не удалось загрузить")

    # проверка совместимости колонок
    base_cols = set(frames[0].columns)
    for i, df in enumerate(frames[1:], start=2):
        if set(df.columns) != base_cols:
            diff = set(df.columns) ^ base_cols
            log.warning(
                "Файл #%d имеет отличающиеся колонки: %s. "
                "Объединение по общим колонкам.",
                i, diff,
            )

    result = pd.concat(frames, ignore_index=True, sort=False)
    log.info("Объединено файлов: %d, отклонено: %d, итог: %d строк",
             loaded, rejected, len(result))
    return result


def iter_chunks(
    path: str | Path,
    chunk_size: int = 10_000,
    **kwargs,
) -> Iterator[pd.DataFrame]:
    """Генератор частей файла (для потоковой обработки). CSV/TXT/XLSX."""
    p = Path(path)
    if not p.exists():
        raise LoadError(f"Файл не найден: {p}")

    fmt = _detect_format(p)
    encoding = kwargs.get("encoding", "utf-8")

    if fmt in {"csv", "txt"}:
        sep = kwargs.get("delimiter") or _detect_delimiter(p, encoding)
        if fmt == "txt" and kwargs.get("delimiter") is None:
            sep = "\t"
        for chunk in pd.read_csv(
            p, sep=sep, encoding=encoding,
            chunksize=chunk_size, header=0,
        ):
            yield chunk
    elif fmt == "xlsx":
        # xlsx читаем целиком и режем вручную (openpyxl не умеет chunks)
        df = pd.read_excel(p, engine="openpyxl")
        for start in range(0, len(df), chunk_size):
            yield df.iloc[start:start + chunk_size].reset_index(drop=True)
    elif fmt == "json":
        df = load_file(p, **kwargs)
        for start in range(0, len(df), chunk_size):
            yield df.iloc[start:start + chunk_size].reset_index(drop=True)
    else:
        raise UnsupportedFormatError(f"Формат {fmt} не поддерживает потоковое чтение")


class ChunkIterator:
    """Собственный итератор (протокол __iter__/__next__) поверх iter_chunks."""

    def __init__(self, path: str | Path, chunk_size: int = 10_000, **kwargs) -> None:
        self._gen = iter_chunks(path, chunk_size=chunk_size, **kwargs)
        self._count = 0

    def __iter__(self) -> "ChunkIterator":
        return self

    def __next__(self) -> pd.DataFrame:
        chunk = next(self._gen)
        self._count += 1
        return chunk

    @property
    def chunks_read(self) -> int:
        return self._count