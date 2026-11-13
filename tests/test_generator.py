"""Тесты генератора: воспроизводимость по seed, порча (nulls/dups/invalid)."""

from modules.generator import generate_dataset


def test_seed_reproducible() -> None:
    a = generate_dataset(rows=50, cols=5, seed=1)
    b = generate_dataset(rows=50, cols=5, seed=1)
    assert a.equals(b)


def test_nulls_injected() -> None:
    df = generate_dataset(rows=100, cols=5, null_pct=0.2, seed=2)
    assert df.isna().sum().sum() > 0


def test_duplicates_injected() -> None:
    df = generate_dataset(rows=100, cols=5, dup_pct=0.3, seed=3)
    assert len(df) > 100


def test_invalid_injected() -> None:
    df = generate_dataset(rows=100, cols=5, invalid_pct=0.1, seed=4)
    assert df["value_0"].dtype == object
    assert "not_a_number" in df["value_0"].astype(str).values