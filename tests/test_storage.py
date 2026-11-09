from modules.storage import Storage


def test_settings_roundtrip(tmp_db: Storage) -> None:
    tmp_db.set_setting("theme", "dark")
    assert tmp_db.get_setting("theme") == "dark"
    assert tmp_db.get_setting("missing", "default") == "default"


def test_scenario_roundtrip(tmp_db: Storage) -> None:
    steps = [{"op": "drop_duplicates", "params": {}}]
    tmp_db.save_scenario("clean_v1", steps)
    assert tmp_db.load_scenario("clean_v1") == steps
    assert "clean_v1" in tmp_db.list_scenarios()


def test_history(tmp_db: Storage) -> None:
    hid = tmp_db.add_history(
        started_at="2026-10-06T10:00:00",
        status="ok",
        input_rows=100,
        output_rows=95,
    )
    assert hid > 0
    rows = tmp_db.list_history()
    assert rows[0]["input_rows"] == 100