"""DataForge — консольный интерфейс.

Два режима:
  python main.py                    # интерактивное меню
  python main.py <команда> [args]   # CLI для скриптов
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


# ============================================================
#  КОМАНДЫ (используются и в CLI, и в интерактивном меню)
# ============================================================

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
    if args.summary:
        print("\nSUMMARY:")
        print(summary(df).to_string())


def cmd_validate(args) -> None:
    df = load_file(args.file)
    result = validate(df, ValidationRules())
    _print_json(result)


def cmd_clean(args) -> None:
    df = load_file(args.file)
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8")) if args.rules else {}
    if not rules:
        print("Правила не переданы — данные не изменятся (только копия в out).")
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
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if dt_cols:
        result["temporal"] = {
            col: temporal_stats(df, col, freq="D").to_dict(orient="records")
            for col in dt_cols
        }
    _print_json(result)


def cmd_temporal(args) -> None:
    df = load_file(args.file)
    out = temporal_stats(df, args.column, freq=args.freq)
    print(out.to_string(index=False))


def cmd_anomalies(args) -> None:
    df = load_file(args.file)
    if args.method == "iqr":
        found = detect_iqr(df, args.column)
    elif args.method == "zscore":
        found = detect_zscore(df, args.column)
    else:
        found = detect_bounds(df, args.column, low=args.low, high=args.high)

    if args.out:
        save_report({"count": len(found), "anomalies": found}, args.out)
        print(f"Найдено {len(found)} аномалий → {args.out}")
    else:
        _print_json({"count": len(found), "anomalies": found})


def cmd_export(args) -> None:
    df = load_file(args.file)
    cols = args.columns.split(",") if args.columns else None
    export(df, args.out, columns=cols, fmt=args.fmt)
    print(f"Экспортировано {len(df)} строк → {args.out}")


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
    files = [str(p) for p in data_dir.glob("*.csv") if p.stat().st_size > 0][:4]
    if not files:
        print("Нет CSV-файлов в data/. Сначала: python main.py generate ...")
        return
    stream_file = files[0]
    df = load_file(stream_file)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        print("В файле нет числовых колонок для бенчмарка")
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


# ============================================================
#  CLI (argparse)
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    epilog = """
Примеры:
  python main.py generate --rows 1000 --nulls 0.1 --out data/gen.csv
  python main.py inspect data/gen.csv --head 5 --summary
  python main.py validate data/gen.csv
  python main.py clean data/gen.csv --rules rules.json --out processed/clean.csv
  python main.py analyze processed/clean.csv
  python main.py temporal processed/clean.csv --column date --freq M
  python main.py anomalies processed/clean.csv --column value_1 --method iqr
  python main.py export processed/clean.csv --out out.json --fmt json
  python main.py scenario-save --name demo --steps steps.json
  python main.py scenario-run --scenario demo --input processed/clean.csv
  python main.py benchmark
  python main.py history
  python main.py            (интерактивное меню)
"""
    p = argparse.ArgumentParser(
        prog="dataforge",
        description="DataForge — система обработки и анализа данных",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

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
    i.add_argument("--summary", action="store_true")
    i.set_defaults(func=cmd_inspect)

    v = sub.add_parser("validate", help="Проверить качество данных")
    v.add_argument("file")
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("clean", help="Очистить данные по JSON-правилам")
    c.add_argument("file")
    c.add_argument("--rules", help="Путь к JSON с правилами")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_clean)

    a = sub.add_parser("analyze", help="Аналитика")
    a.add_argument("file")
    a.set_defaults(func=cmd_analyze)

    t = sub.add_parser("temporal", help="Анализ по временным интервалам")
    t.add_argument("file")
    t.add_argument("--column", required=True)
    t.add_argument("--freq", default="D", help="D/W/ME/YE")
    t.set_defaults(func=cmd_temporal)

    an = sub.add_parser("anomalies", help="Поиск аномалий")
    an.add_argument("file")
    an.add_argument("--column", required=True)
    an.add_argument("--method", choices=["iqr", "zscore", "bounds"], default="iqr")
    an.add_argument("--low", type=float, default=0.0)
    an.add_argument("--high", type=float, default=100.0)
    an.add_argument("--out", help="Сохранить полный список в JSON")
    an.set_defaults(func=cmd_anomalies)

    e = sub.add_parser("export", help="Экспорт данных в CSV/JSON/XLSX")
    e.add_argument("file")
    e.add_argument("--out", required=True)
    e.add_argument("--columns", help="Колонки через запятую")
    e.add_argument("--fmt", choices=["csv", "json", "xlsx"])
    e.set_defaults(func=cmd_export)

    ss = sub.add_parser("scenario-save", help="Сохранить сценарий из JSON")
    ss.add_argument("--name", required=True)
    ss.add_argument("--steps", required=True)
    ss.set_defaults(func=cmd_scenario_save)

    sr = sub.add_parser("scenario-run", help="Запустить сохранённый сценарий")
    sr.add_argument("--scenario", required=True)
    sr.add_argument("--input", required=True)
    sr.add_argument("--out")
    sr.set_defaults(func=cmd_scenario_run)

    b = sub.add_parser("benchmark", help="Запустить все сравнения")
    b.set_defaults(func=cmd_benchmark)

    h = sub.add_parser("history", help="История запусков")
    h.add_argument("--limit", type=int, default=20)
    h.set_defaults(func=cmd_history)

    return p


def run_cli(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        args.func(args)
    except DataForgeError as e:
        log.error("Ошибка: %s", e)
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 0


# ============================================================
#  Интерактивное меню
# ============================================================

def _ask(prompt: str, default: str = "", required: bool = False) -> str:
    raw = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    if not raw:
        if default:
            return default
        if required:
            raise DataForgeError(f"Поле обязательно: {prompt}")
        return ""
    return raw


def _pick_file(prompt: str) -> str:
    cfg = load_config()
    candidates: list[Path] = []
    for d in (cfg.data_dir, cfg.processed_dir):
        p = Path(d)
        if p.exists():
            for ext in ("*.csv", "*.json", "*.xlsx"):
                candidates.extend(sorted(p.glob(ext)))

    if candidates:
        print("\nДоступные файлы:")
        for i, f in enumerate(candidates, 1):
            print(f"  {i}) {f}")
        print("  или введите путь вручную")
    else:
        print("\nНет файлов в data/ и processed/. Сначала: generate.")

    raw = input(f"{prompt} (номер или путь): ").strip()
    if not raw:
        raise DataForgeError("Путь не введён")
    if raw.isdigit():
        idx = int(raw) - 1
        if not (0 <= idx < len(candidates)):
            raise DataForgeError(f"Нет файла с номером {raw}")
        return str(candidates[idx])
    return raw


def _pick_column(file: str, prompt: str, numeric_only: bool = False) -> str:
    """Показывает колонки файла и даёт выбрать номером или именем."""
    try:
        df = load_file(file)
    except DataForgeError:
        raise
    cols = list(df.columns)
    if numeric_only:
        cols = df.select_dtypes(include="number").columns.tolist()
        if not cols:
            raise DataForgeError("В файле нет числовых колонок")
    if not cols:
        raise DataForgeError("В файле нет колонок")

    print(f"\nКолонки в {file}:")
    for i, c in enumerate(cols, 1):
        print(f"  {i}) {c}")
    raw = input(f"{prompt} (номер или имя): ").strip()
    if not raw:
        raise DataForgeError("Колонка не введена")
    if raw.isdigit():
        idx = int(raw) - 1
        if not (0 <= idx < len(cols)):
            raise DataForgeError(f"Нет колонки с номером {raw}")
        return cols[idx]
    if raw not in cols:
        raise DataForgeError(f"Колонка '{raw}' не найдена. Доступно: {cols}")
    return raw


def _pause() -> None:
    input("\nНажмите Enter, чтобы вернуться в меню...")


def _menu_generate() -> None:
    rows = int(_ask("Сколько строк", "500"))
    nulls = float(_ask("Доля пропусков (0..1)", "0.1"))
    dups = float(_ask("Доля дубликатов (0..1)", "0.1"))
    invalid = float(_ask("Доля некорректных (0..1)", "0.05"))
    out = _ask("Куда сохранить", "data/gen.csv")
    ns = argparse.Namespace(
        rows=rows, cols=5, categories=5,
        nulls=nulls, dups=dups, invalid=invalid, seed=42, out=out,
    )
    cmd_generate(ns)


def _menu_inspect() -> None:
    f = _pick_file("Какой файл открыть")
    ns = argparse.Namespace(file=f, head=5, summary=True)
    cmd_inspect(ns)


def _menu_validate() -> None:
    f = _pick_file("Какой файл проверить")
    cmd_validate(argparse.Namespace(file=f))


def _menu_clean() -> None:
    f = _pick_file("Какой файл очистить")
    rules_path = _ask("Путь к JSON с правилами (Enter — без правил)", "")
    out = _ask("Куда сохранить результат", "processed/clean.csv")
    ns = argparse.Namespace(file=f, rules=rules_path or None, out=out)
    cmd_clean(ns)


def _menu_analyze() -> None:
    f = _pick_file("Какой файл проанализировать")
    cmd_analyze(argparse.Namespace(file=f))


def _menu_temporal() -> None:
    f = _pick_file("Какой файл")
    col = _pick_column(f, "Колонка с датой")
    freq = _ask("Интервал (D/W/ME/YE)", "D")
    cmd_temporal(argparse.Namespace(file=f, column=col, freq=freq))


def _menu_anomalies() -> None:
    f = _pick_file("Какой файл")
    col = _pick_column(f, "Колонка для поиска", numeric_only=True)
    method = _ask("Метод (iqr/zscore/bounds)", "iqr")
    out = _ask("Куда сохранить результат", "processed/anomalies.json")
    ns = argparse.Namespace(
        file=f, column=col, method=method,
        low=0.0, high=1000.0, out=out or None,
    )
    cmd_anomalies(ns)


def _menu_export() -> None:
    f = _pick_file("Какой файл экспортировать")
    out = _ask("Куда сохранить", "processed/out.json")
    cols = _ask("Колонки через запятую (Enter — все)", "")
    fmt = _ask("Формат (csv/json/xlsx, Enter — по расширению)", "")
    cmd_export(argparse.Namespace(
        file=f, out=out,
        columns=cols or None,
        fmt=fmt or None,
    ))


def _menu_scenario_save() -> None:
    name = _ask("Имя сценария", required=True)
    steps_path = _ask("Путь к JSON с шагами", "steps.json")
    p = Path(steps_path)
    if not p.exists():
        print(f"\nФайл {steps_path} не найден.")
        create = input("Создать шаблон? (y/n): ").strip().lower()
        if create != "y":
            print("Отменено.")
            return
        template = [
            {"op": "add_computed", "params": {"name": "new_col", "expression": "value_1 * 2"}},
            {"op": "sort_df", "params": {"by": ["new_col"], "ascending": False}},
        ]
        p.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Создан шаблон {steps_path}")
    cmd_scenario_save(argparse.Namespace(name=name, steps=steps_path))


def _menu_scenario_run() -> None:
    name = _ask("Имя сценария", required=True)
    f = _pick_file("Входной файл")
    out = _ask("Куда сохранить результат (Enter — не сохранять)", "")
    cmd_scenario_run(argparse.Namespace(
        scenario=name, input=f, out=out or None,
    ))


def _menu_benchmark() -> None:
    cmd_benchmark(argparse.Namespace())


def _menu_history() -> None:
    cmd_history(argparse.Namespace(limit=20))


def _menu_help() -> None:
    build_parser().print_help()


_MENU = {
    "1":  ("Сгенерировать тестовые данные",          _menu_generate),
    "2":  ("Показать структуру файла (inspect)",     _menu_inspect),
    "3":  ("Проверить качество (validate)",          _menu_validate),
    "4":  ("Очистить данные (clean)",                _menu_clean),
    "5":  ("Аналитика (analyze)",                    _menu_analyze),
    "6":  ("Временной анализ (temporal)",            _menu_temporal),
    "7":  ("Поиск аномалий (anomalies)",             _menu_anomalies),
    "8":  ("Экспорт (export)",                       _menu_export),
    "9":  ("Сохранить сценарий (scenario-save)",     _menu_scenario_save),
    "10": ("Запустить сценарий (scenario-run)",      _menu_scenario_run),
    "11": ("Benchmark",                              _menu_benchmark),
    "12": ("История запусков",                       _menu_history),
    "13": ("Справка по CLI",                         _menu_help),
    "0":  ("Выход",                                  None),
}


def run_interactive() -> int:
    while True:
        print()
        print("=== DataForge ===")
        print("Выберите действие:")
        for key, (title, _) in _MENU.items():
            print(f"  {key:>2}) {title}")
        choice = input("\nВаш выбор: ").strip()
        if choice in ("0", "q", "quit", "exit"):
            print("Пока!")
            return 0
        item = _MENU.get(choice)
        if not item or item[1] is None:
            print("Нет такого пункта. Попробуйте снова.")
            continue
        try:
            item[1]()
        except DataForgeError as e:
            print(f"Ошибка: {e}")
        except (KeyboardInterrupt, EOFError):
            print("\nПрерывание. Выход.")
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"Ошибка выполнения: {e}")
        _pause()


# ============================================================
#  Точка входа
# ============================================================

def main(argv: list[str] | None = None) -> int:
    log.info("DataForge старт")
    argv = argv if argv is not None else sys.argv[1:]
    try:
        if argv:
            return run_cli(argv)
        return run_interactive()
    finally:
        log.info("DataForge завершение")


if __name__ == "__main__":
    raise SystemExit(main())