"""Unit tests for API health endpoint and application lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from flight_delay.api.main import create_app
from flight_delay.monitoring.metrics import FlightDelayMetrics


def test_health_when_fully_healthy(monkeypatch) -> None:
    """Test /health returns 'ok' when model is ready and Cassandra is connected."""
    mock_predictor = MagicMock()
    mock_predictor._pipeline = MagicMock()

    mock_storage = MagicMock()
    mock_storage.is_connected = True

    metrics = FlightDelayMetrics()
    app = create_app(predictor=mock_predictor, storage=mock_storage, metrics=metrics)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_ready"] is True
        assert data["cassandra_connected"] is True
        assert data["version"] == "0.1.0"
        # Ensure no internal credentials leaked
        assert "password" not in data
        assert "secret" not in data
        assert "keyspace" not in data


def test_health_when_cassandra_offline_returns_degraded() -> None:
    """Test /health returns 'degraded' when model is ready but Cassandra is disconnected."""
    mock_predictor = MagicMock()
    mock_predictor._pipeline = MagicMock()

    mock_storage = MagicMock()
    mock_storage.is_connected = False

    metrics = FlightDelayMetrics()
    app = create_app(predictor=mock_predictor, storage=mock_storage, metrics=metrics)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_ready"] is True
        assert data["cassandra_connected"] is False


def test_health_when_model_not_ready_returns_unhealthy() -> None:
    """Test /health returns 'unhealthy' when model is not ready."""
    mock_storage = MagicMock()
    mock_storage.is_connected = True

    metrics = FlightDelayMetrics()
    app = create_app(predictor=None, storage=mock_storage, metrics=metrics)
    # Ensure app.state.predictor is explicitly None
    app.state.predictor = None

    with TestClient(app) as client:
        # Override to keep predictor None
        app.state.predictor = None
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_ready"] is False
