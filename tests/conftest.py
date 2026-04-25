"""Shared pytest fixtures: clear global settings cache between cases."""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings


def settings_for_tests(**overrides: object) -> Settings:
    """
    Build :class:`Settings` without running env/``.env`` resolution (tests only).
    Unset fields use model defaults.
    """
    return Settings.model_construct(**overrides)  # type: ignore[call-arg, return-value]


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
