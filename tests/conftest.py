import pytest
from pathlib import Path
from modules.storage import Storage
from config import AppConfig


@pytest.fixture
def tmp_db(tmp_path: Path) -> Storage:
    st = Storage(tmp_path / "test.db")
    yield st
    st.close()


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()