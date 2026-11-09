"""Benchmark: 4 сравнения из ТЗ + сохранение результатов."""
import asyncio
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from modules.loader import iter_chunks, load_file
from modules.anomaly_detector import detect_iqr
from utils.exceptions import BenchmarkError
from utils.logger import get_logger

log = get_logger("benchmark")


def time_it(fn: Callable, *args, **kwargs) -> tuple[float, object]:
    """Возвращает (duration_sec, result)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


# ========== 1. Python loops vs NumPy ==========

def zscore_python(values: list[float]) -> list[float]:
    """Z-score через чистый Python."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = var ** 0.5
    if std == 0:
        return [0.0] * n
    return [(x - mean) / std for x in values]


def zscore_numpy(values: np.ndarray) -> np.ndarray:
    """Z-score через NumPy (векторизация)."""
    mean = values.mean()
    std = values.std()
    if std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def bench_numpy_vs_loops(n: int = 200_000, seed: int = 42) -> dict:
    """Сравнение циклов и векторизации на z-score."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=n)
    lst = arr.tolist()

    t_py, _ = time_it(zscore_python, lst)
    t_np, _ = time_it(zscore_numpy, arr)

    result = {
        "task": "z-score",
        "n": n,
        "python_loops_sec": t_py,
        "numpy_sec": t_np,
        "speedup": t_py / t_np if t_np > 0 else None,
        "why_vectorized": (
            "z-score — поэлементная арифметика над однородным массивом. "
            "NumPy выполняет её в C без создания Python-объектов на каждом шаге, "
            "поэтому операция векторизуется."
        ),
    }
    log.info("NumPy vs loops: %.4fs vs %.4fs (x%.1f)",
             t_py, t_np, result["speedup"] or 0)
    return result


# ========== 2. Обычная vs потоковая обработка ==========

def _streaming_stats(path: str, chunk_size: int) -> dict:
    """Потоковая статистика: count, sum, mean, sum_sq, min, max."""
    count = 0
    total = 0.0
    total_sq = 0.0
    vmin = float("inf")
    vmax = float("-inf")

    for chunk in iter_chunks(path, chunk_size=chunk_size):
        numeric = chunk.select_dtypes(include="number")
        if numeric.empty:
            continue
        for col in numeric.columns:
            s = numeric[col].dropna()
            if s.empty:
                continue
            count += int(s.count())
            total += float(s.sum())
            total_sq += float((s ** 2).sum())
            vmin = min(vmin, float(s.min()))
            vmax = max(vmax, float(s.max()))

    return {
        "count": count,
        "sum": total,
        "sum_sq": total_sq,
        "min": vmin if count else None,
        "max": vmax if count else None,
        "mean": (total / count) if count else None,
    }


def _inmemory_stats(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": None, "max": None, "mean": None}
    s = numeric.stack().dropna()
    return {
        "count": int(s.count()),
        "sum": float(s.sum()),
        "sum_sq": float((s ** 2).sum()),
        "min": float(s.min()) if len(s) else None,
        "max": float(s.max()) if len(s) else None,
        "mean": float(s.mean()) if len(s) else None,
    }


def bench_streaming_vs_memory(path: str, chunk_size: int = 1000) -> dict:
    """Сверяет результаты потоковой и обычной обработки + замеряет время."""
    try:
        df = load_file(path)
    except Exception as e:
        raise BenchmarkError(f"Не удалось загрузить {path}: {e}") from e

    t_mem, mem = time_it(_inmemory_stats, df)
    t_str, stream = time_it(_streaming_stats, path, chunk_size)

    # сверка (раздел 15 ТЗ) — результаты должны совпадать
    mismatch = []
    for key in ("count", "min", "max"):
        if mem[key] != stream[key]:
            mismatch.append({"key": key, "memory": mem[key], "stream": stream[key]})
    for key in ("sum", "sum_sq", "mean"):
        mv, sv = mem[key], stream[key]
        if mv is None or sv is None:
            continue
        if abs(mv - sv) > 1e-6 * max(1.0, abs(mv)):
            mismatch.append({"key": key, "memory": mv, "stream": sv})

    result = {
        "task": "streaming_vs_memory",
        "file": str(path),
        "chunk_size": chunk_size,
        "rows": int(len(df)),
        "memory_sec": t_mem,
        "streaming_sec": t_str,
        "matched": len(mismatch) == 0,
        "mismatch": mismatch,
    }
    log.info("Streaming vs memory: matched=%s", result["matched"])
    return result


# ========== 3. Последовательная vs многопоточная обработка ==========

def _read_and_count(path: str) -> int:
    df = load_file(path)
    return int(len(df))


def bench_threads_vs_seq(files: list[str], n_threads: int = 4) -> dict:
    """Чтение и подготовка нескольких файлов: последовательно vs threads."""
    if not files:
        raise BenchmarkError("Нет файлов для бенчмарка")

    t_seq, seq_counts = time_it(lambda: [_read_and_count(f) for f in files])

    def _run_threads() -> list[int]:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            return list(ex.map(_read_and_count, files))

    t_thr, thr_counts = time_it(_run_threads)

    result = {
        "task": "threads_vs_seq",
        "n_files": len(files),
        "n_threads": n_threads,
        "seq_sec": t_seq,
        "threads_sec": t_thr,
        "speedup": t_seq / t_thr if t_thr > 0 else None,
        "matched": seq_counts == thr_counts,
        "why_threads": (
            "Чтение файлов — I/O-bound: большую часть времени поток ждёт диск/сеть. "
            "GIL отпускается на время I/O, поэтому threads дают выигрыш. "
            "Для CPU-bound операций threads не помогут из-за GIL."
        ),
    }
    log.info("Threads vs seq: %.4fs vs %.4fs", t_seq, t_thr)
    return result


# ========== 4. Последовательная vs многопроцессорная обработка ==========

def _process_chunk_iqr(args: tuple[pd.DataFrame, str]) -> int:
    """CPU-bound операция над частью: количество аномалий по IQR в колонке."""
    chunk, column = args
    return len(detect_iqr(chunk, column))


def bench_processes_vs_seq(
    df: pd.DataFrame, column: str, n_processes: int = 4, n_parts: int = 4
) -> dict:
    """Поиск аномалий по частям: последовательно vs multiprocessing."""
    if column not in df.columns:
        raise BenchmarkError(f"Колонка не найдена: {column}")

    parts = [df.iloc[i::n_parts].reset_index(drop=True) for i in range(n_parts)]
    args = [(p, column) for p in parts]

    t_seq, seq_counts = time_it(lambda: [_process_chunk_iqr(a) for a in args])

    def _run_procs() -> list[int]:
        with ProcessPoolExecutor(max_workers=n_processes) as ex:
            return list(ex.map(_process_chunk_iqr, args))

    t_prc, prc_counts = time_it(_run_procs)

    total_seq = sum(seq_counts)
    total_prc = sum(prc_counts)

    result = {
        "task": "processes_vs_seq",
        "column": column,
        "rows": int(len(df)),
        "n_parts": n_parts,
        "n_processes": n_processes,
        "seq_sec": t_seq,
        "processes_sec": t_prc,
        "speedup": t_seq / t_prc if t_prc > 0 else None,
        "total_anomalies_seq": total_seq,
        "total_anomalies_proc": total_prc,
        "matched": total_seq == total_prc,
        "why_processes": (
            "Поиск аномалий по IQR — CPU-bound. GIL не даёт threads ускорить "
            "вычисления, поэтому нужны процессы. Части независимы, "
            "поэтому результат собирается без потерь и дублей."
        ),
    }
    log.info("Processes vs seq: %.4fs vs %.4fs, matched=%s",
             t_seq, t_prc, result["matched"])
    return result


# ========== 5. asyncio — конкурентное чтение файлов ==========

async def _async_read(path: str) -> int:
    """Корутина: блокирующее чтение через executor, чтобы не блокировать loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _read_and_count, path)


async def _async_read_all(files: list[str]) -> list[int]:
    return await asyncio.gather(*[_async_read(f) for f in files])


def bench_asyncio(files: list[str]) -> dict:
    """Сравнение последовательного и асинхронного чтения нескольких файлов."""
    if not files:
        raise BenchmarkError("Нет файлов для asyncio-бенчмарка")

    t_seq, seq_counts = time_it(lambda: [_read_and_count(f) for f in files])
    t_async, async_counts = time_it(lambda: asyncio.run(_async_read_all(files)))

    result = {
        "task": "asyncio_vs_seq",
        "n_files": len(files),
        "seq_sec": t_seq,
        "async_sec": t_async,
        "matched": seq_counts == async_counts,
        "why_asyncio": (
            "Чтение нескольких файлов — I/O-bound. asyncio позволяет запустить "
            "несколько корутин одновременно и переключаться между ними, пока одна "
            "ждёт диск. Блокирующий pd.read_csv оборачивается в run_in_executor, "
            "чтобы не блокировать event loop. asyncio не ускоряет CPU-вычисления."
        ),
    }
    log.info("Async vs seq: %.4fs vs %.4fs", t_seq, t_async)
    return result


# ========== Сохранение результатов ==========

def save_benchmark(results: dict, path: str | Path) -> None:
    """Сохраняет результаты в JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        log.info("Benchmark сохранён: %s", p)
    except OSError as e:
        raise BenchmarkError(f"Не удалось сохранить benchmark: {e}") from e


def run_all(
    files: list[str],
    stream_file: str,
    df_for_processes: pd.DataFrame,
    process_column: str,
    out_path: str | Path,
) -> dict:
    """Запускает все 4 сравнения + asyncio и сохраняет отчёт."""
    results = {
        "numpy_vs_loops": bench_numpy_vs_loops(),
        "streaming_vs_memory": bench_streaming_vs_memory(stream_file, chunk_size=1000),
        "threads_vs_seq": bench_threads_vs_seq(files, n_threads=4),
        "processes_vs_seq": bench_processes_vs_seq(
            df_for_processes, process_column, n_processes=4, n_parts=4
        ),
        "asyncio_vs_seq": bench_asyncio(files),
    }
    save_benchmark(results, out_path)
    return results