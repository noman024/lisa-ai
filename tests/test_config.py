"""SQA: application settings and env-based path overrides — app.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, get_settings
from tests.conftest import settings_for_tests


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
    get_settings.cache_clear()


def test_settings_construct_custom_paths(tmp_path: Path) -> None:
    kb = tmp_path / "my_kb.md"
    kb.write_text("## X\n\nbody", encoding="utf-8")
    data = tmp_path / "idx"
    s = settings_for_tests(
        data_dir=data,
        knowledge_path=kb,
        llm_base_url="http://127.0.0.1:1/v1",
        llm_model="m",
    )
    assert s.data_dir == data
    assert s.knowledge_path == kb
    read_back = s.knowledge_path.read_text(encoding="utf-8")
    assert "## X" in read_back


def test_knowledge_path_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb = tmp_path / "from_env.md"
    kb.write_text("## Y\n\ntest", encoding="utf-8")
    monkeypatch.setenv("KNOWLEDGE_PATH", str(kb))
    get_settings.cache_clear()
    try:
        s = Settings()  # fresh from env, not cached old singleton
    finally:
        get_settings.cache_clear()
    assert s.knowledge_path == kb


def test_data_dir_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "data_alias"
    d.mkdir()
    monkeypatch.setenv("DATA_DIR", str(d))
    get_settings.cache_clear()
    try:
        s = Settings()
    finally:
        get_settings.cache_clear()
    assert s.data_dir == d
