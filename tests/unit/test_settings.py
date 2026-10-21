"""Tests for environment-backed application settings."""

import pytest
from pydantic import ValidationError

from flight_delay.config.settings import Settings


def test_settings_load_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("CASSANDRA_PORT", "9142")

    settings = Settings(_env_file=None)

    assert settings.app_env == "test"
    assert settings.cassandra_port == 9142


def test_aviationstack_key_is_required_only_when_source_is_used() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ValueError, match="AVIATIONSTACK_API_KEY is required"):
        settings.require_aviationstack_api_key()


def test_invalid_cassandra_port_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASSANDRA_PORT", "70000")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
