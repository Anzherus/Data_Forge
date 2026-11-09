"""DataForge — консольный интерфейс.

Примеры:
    python main.py generate --rows 1000 --nulls 0.1 --out data/gen.csv
    python main.py inspect data/gen.csv
    python main.py validate data/gen.csv
    python main.py clean data/gen.csv --rules configs/clean.json --out processed/clean.csv
    python main.py scenario-run --scenario clean_v1 --input data/gen.csv
    python main.py benchmark
"""
import argparse
import json
import sys
from pathlib import Path

from config import load_config
from modules.analyzer import (
    categorical_stats,
    correlations,
    numeric_stats,
    temporal_stats,
)
from modules.anomaly_detector import detect_bounds, detect_iqr, detect_zscore
from modules.benchmark import run_all
from modules.cleaner import apply_cleaning
from modules.exporter import export, save_report
from modules.generator import generate_dataset
from modules.inspector import head_tail, inspect, summary
from modules.loader import load_file
from modules.pipeline import Step, load_scenario, run_pipeline, save_scenario
from modules.storage import Storage
from modules.validator import ValidationRules, validate
from utils.exceptions import DataForgeError
from utils.logger import get_logger

log = get_logger("main")


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


# ---------- команды ----------

def cmd_generate(args) -> None:
    df = generate_dataset(
        rows=args.rows,
        cols=args.cols,
        categories=args.categories,
        null_pct=args.nulls,
        dup_pct=args.dups,
        invalid_pct=args.invalid,
        seed=args.seed,
    )
    export(df, args.out)
    print(f"Сгенерировано {len(df)} строк → {args.out}")


def cmd_inspect(args) -> None:
    df = load_file(args.file)
    info = inspect(df)
    _print_json(info)
    if args.head:
        print("\nHEAD:")
        print(head_tail(df, n=args.head)["head"].to_string(index=False))


def cmd_validate(args) -> None:
    df = load_file(args.file)
    result = validate(df, ValidationRules())
    _print_json(result)


def cmd_clean(args) -> None:
    df = load_file(args.file)
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8")) if args.rules else {}
    out, report = apply_cleaning(df, rules)
    export(out, args.out)
    save_report(report, Path(args.out).with_suffix(".report.json"))
    _print_json(report)


def cmd_analyze(args) -> None:
    df = load_file(args.file)
    result = {
        "numeric": numeric_stats(df),
        "categorical": categorical_stats(df),
        "correlations": correlations(df),
    }
    _print_json(result)


def cmd_anomalies(args) -> None:
    df = load_file(args.file)
    method = args.method
    if method == "iqr":
        found = detect_iqr(df, args.column)
    elif method == "zscore":
        found = detect_zscore(df, args.column)
    else:
        found = detect_bounds(df, args.column, low=args.low, high=args.high)
    _print_json({"count": len(found), "anomalies": found[:50]})


def cmd_scenario_save(args) -> None:
    steps = json.loads(Path(args.steps).read_text(encoding="utf-8"))
    step_objs = [Step(op=s["op"], params=s.get("params", {})) for s in steps]
    cfg = load_config()
    storage = Storage(cfg.db_path)
    try:
        save_scenario(storage, args.name, step_objs)
        print(f"Сценарий '{args.name}' сохранён ({len(step_objs)} шагов)")
    finally:
        storage.close()


def cmd_scenario_run(args) -> None:
    df = load_file(args.input)
    cfg = load_config()
    storage = Storage(cfg.db_path)
    try:
        steps = load_scenario(storage, args.scenario)
        out, history = run_pipeline(df, steps, storage=storage, scenario_name=args.scenario)
        if args.out:
            export(out, args.out)
        _print_json(history)
    finally:
        storage.close()


def cmd_benchmark(args) -> None:
    cfg = load_config()
    data_dir = Path(cfg.data_dir)
    files = [str(p) for p in data_dir.glob("*.csv")][:4]
    if not files:
        print("Нет CSV-файлов в data/ для бенчмарка. Сначала: python main.py generate ...")
        return

    stream_file = files[0]
    df = load_file(stream_file)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        print("В файле нет числовых колонок для процесса-бенчмарка")
        return

    out_path = Path(cfg.reports_dir) / "benchmark.json"
    run_all(
        files=files,
        stream_file=stream_file,
        df_for_processes=df,
        process_column=numeric_cols[0],
        out_path=out_path,
    )
    print(f"Benchmark сохранён: {out_path}")


def cmd_history(args) -> None:
    cfg = load_config()
    storage = Storage(cfg.db_path)
    try:
        rows = storage.list_history(limit=args.limit)
        _print_json(rows)
    finally:
        storage.close()


# ---------- CLI ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dataforge", description="DataForge CLI")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Сгенерировать тестовые данные")
    g.add_argument("--rows", type=int, default=1000)
    g.add_argument("--cols", type=int, default=5)
    g.add_argument("--categories", type=int, default=5)
    g.add_argument("--nulls", type=float, default=0.0)
    g.add_argument("--dups", type=float, default=0.0)
    g.add_argument("--invalid", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_generate)

    i = sub.add_parser("inspect", help="Показать структуру файла")
    i.add_argument("file")
    i.add_argument("--head", type=int, default=0)
    i.set_defaults(func=cmd_inspect)

    v = sub.add_parser("validate", help="Проверить качество данных")
    v.add_argument("file")
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("clean", help="Очистить данные по JSON-правилам")
    c.add_argument("file")
    c.add_argument("--rules", help="Путь к JSON с правилами")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_clean)

    a = sub.add_parser("analyze", help="Аналитика: числовая, категориальная, корреляции")
    a.add_argument("file")
    a.set_defaults(func=cmd_analyze)

    an = sub.add_parser("anomalies", help="Поиск аномалий")
    an.add_argument("file")
    an.add_argument("--column", required=True)
    an.add_argument("--method", choices=["iqr", "zscore", "bounds"], default="iqr")
    an.add_argument("--low", type=float, default=0.0)
    an.add_argument("--high", type=float, default=100.0)
    an.set_defaults(func=cmd_anomalies)

    ss = sub.add_parser("scenario-save", help="Сохранить сценарий из JSON")
    ss.add_argument("--name", required=True)
    ss.add_argument("--steps", required=True)
    ss.set_defaults(func=cmd_scenario_save)

    sr = sub.add_parser("scenario-run", help="Запустить сохранённый сценарий")
    sr.add_argument("--scenario", required=True)
    sr.add_argument("--input", required=True)
    sr.add_argument("--out")
    sr.set_defaults(func=cmd_scenario_run)

    b = sub.add_parser("benchmark", help="Запустить все 4 сравнения")
    b.set_defaults(func=cmd_benchmark)

    h = sub.add_parser("history", help="История запусков")
    h.add_argument("--limit", type=int, default=20)
    h.set_defaults(func=cmd_history)

    return p


def main(argv: list[str] | None = None) -> int:
    log.info("DataForge старт")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except DataForgeError as e:
        log.error("Ошибка: %s", e)
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    log.info("DataForge завершение")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())