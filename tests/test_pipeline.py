"""Тесты сценариев: выполнение, ошибка с номером шага, сохранение/загрузка, история."""

import pandas as pd
import pytest

from modules.pipeline import Step, load_scenario, run_pipeline, save_scenario
from modules.storage import Storage
from utils.exceptions import PipelineError


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})


def test_run_pipeline(df) -> None:
    steps = [
        Step("add_computed", {"name": "c", "expression": "a + b"}),
        Step("filter_df", {"expression": "c > 15"}),
    ]
    out, hist = run_pipeline(df, steps)
    assert len(out) == 2
    assert "c" in out.columns
    assert hist["input_rows"] == 3
    assert hist["output_rows"] == 2


def test_pipeline_error_reports_step(df) -> None:
    """При падении сценария должен быть указан номер шага"""
    
    steps = [
        Step("add_computed", {"name": "x", "expression": "a + b"}),
        Step("drop_columns", {"columns": ["nope"]}),
    ]
    with pytest.raises(PipelineError) as exc:
        run_pipeline(df, steps)
    assert "Шаг 2" in str(exc.value)


def test_scenario_roundtrip(df, tmp_db: Storage) -> None:
    steps = [Step("add_computed", {"name": "c", "expression": "a * 2"})]
    save_scenario(tmp_db, "test_sc", steps)
    loaded = load_scenario(tmp_db, "test_sc")
    assert len(loaded) == 1
    assert loaded[0].op == "add_computed"
    out, _ = run_pipeline(df, loaded)
    assert out["c"].tolist() == [2, 4, 6]


def test_pipeline_records_history(df, tmp_db: Storage) -> None:
    steps = [Step("filter_df", {"expression": "a > 1"})]
    run_pipeline(df, steps, storage=tmp_db, scenario_name="h_test")
    rows = tmp_db.list_history()
    assert rows[0]["scenario_name"] == "h_test"
    assert rows[0]["input_rows"] == 3
    assert rows[0]["output_rows"] == 2