"""Unit tests for API airport predictions endpoint."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from flight_delay.api.main import create_app
from flight_delay.monitoring.metrics import FlightDelayMetrics
from flight_delay.storage.cassandra import PredictionRecord


def test_get_airport_predictions_valid() -> None:
    """Test GET /api/v1/airports/{airport}/predictions with valid params."""
    mock_storage = MagicMock()
    mock_storage.is_connected = True

    now = datetime.datetime.now(datetime.UTC)
    mock_record = PredictionRecord(
        event_id="evt_ord_001",
        carrier="UA",
        flight_number="200",
        origin="ORD",
        dest="SFO",
        event_time=now,
        prediction_time=now,
        prediction_date=now.date(),
        delay_probability=0.72,
        predicted_delayed=True,
        model_version="1.0.0",
    )
    mock_storage.get_predictions_by_airport.return_value = [mock_record]

    app = create_app(storage=mock_storage, metrics=FlightDelayMetrics())
    with TestClient(app) as client:
        resp = client.get(
            f"/api/v1/airports/ORD/predictions?date={now.date().isoformat()}&limit=50"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["airport"] == "ORD"
        assert data["date"] == now.date().isoformat()
        assert data["limit"] == 50
        assert data["count"] == 1
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["event_id"] == "evt_ord_001"
        assert data["predictions"][0]["delay_probability"] == 0.72
        assert data["predictions"][0]["predicted_delayed"] is True


def test_get_airport_predictions_invalid_airport_code() -> None:
    """Test GET /api/v1/airports/{airport}/predictions with invalid code returns 422."""
    mock_storage = MagicMock()
    mock_storage.is_connected = True
    app = create_app(storage=mock_storage, metrics=FlightDelayMetrics())

    with TestClient(app) as client:
        # 4 letters
        assert client.get("/api/v1/airports/KORD/predictions").status_code == 422
        # Numbers
        assert client.get("/api/v1/airports/123/predictions").status_code == 422
        # Special chars
        assert client.get("/api/v1/airports/O@D/predictions").status_code == 422


def test_get_airport_predictions_invalid_date_format() -> None:
    """Test GET /api/v1/airports/{airport}/predictions with invalid date returns 422."""
    mock_storage = MagicMock()
    mock_storage.is_connected = True
    app = create_app(storage=mock_storage, metrics=FlightDelayMetrics())

    with TestClient(app) as client:
        assert client.get("/api/v1/airports/ORD/predictions?date=15-06-2026").status_code == 422
        assert client.get("/api/v1/airports/ORD/predictions?date=not-a-date").status_code == 422


def test_get_airport_predictions_limit_bounds() -> None:
    """Test GET /api/v1/airports/{airport}/predictions respects limit bounds [1, 1000]."""
    mock_storage = MagicMock()
    mock_storage.is_connected = True
    app = create_app(storage=mock_storage, metrics=FlightDelayMetrics())

    with TestClient(app) as client:
        # Limit 0 rejected
        assert client.get("/api/v1/airports/ORD/predictions?limit=0").status_code == 422
        # Limit 1001 rejected
        assert client.get("/api/v1/airports/ORD/predictions?limit=1001").status_code == 422


def test_get_airport_predictions_cassandra_offline_returns_503() -> None:
    """Test GET /api/v1/airports/{airport}/predictions returns 503 when Cassandra is down."""
    mock_storage = MagicMock()
    mock_storage.is_connected = False
    app = create_app(storage=mock_storage, metrics=FlightDelayMetrics())

    with TestClient(app) as client:
        resp = client.get("/api/v1/airports/ORD/predictions")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"]
