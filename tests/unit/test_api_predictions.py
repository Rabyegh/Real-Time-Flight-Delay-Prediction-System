"""Unit tests for API prediction endpoints (single, batch, event retrieval)."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from flight_delay.api.main import create_app
from flight_delay.config.settings import Settings
from flight_delay.monitoring.metrics import FlightDelayMetrics
from flight_delay.storage.cassandra import PredictionRecord
from flight_delay.training.inference import ModelPredictor

REPRESENTATIVE_VALID_RECORD = {
    "carrier": "AA",
    "origin": "JFK",
    "dest": "LAX",
    "crs_dep_hour": 9,
    "crs_dep_minute": 30,
    "dep_time_blk": "0900-0959",
    "crs_arr_hour": 12,
    "crs_arr_minute": 45,
    "arr_time_blk": "1200-1259",
    "month": 6,
    "day_of_month": 15,
    "day_of_week": 3,
    "is_weekend": 0,
    "crs_elapsed_time": 375.0,
    "distance": 2475.0,
    "event_id": "test_event_123",
}


def test_predict_single_valid() -> None:
    """Test POST /api/v1/predictions with valid flight record returns prediction."""
    metrics = FlightDelayMetrics()
    predictor = ModelPredictor(threshold=0.5, model_version="1.0.0")
    app = create_app(predictor=predictor, metrics=metrics)

    with TestClient(app) as client:
        resp = client.post("/api/v1/predictions", json=REPRESENTATIVE_VALID_RECORD)
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["delay_probability"] <= 1.0
        assert isinstance(data["predicted_delayed"], bool)
        assert isinstance(data["is_delayed"], bool)
        assert data["threshold"] == 0.5
        assert data["model_version"] == "1.0.0"
        assert data["event_id"] == "test_event_123"
        assert "prediction_time" in data


def test_predict_single_missing_feature_rejected() -> None:
    """Test POST /api/v1/predictions with missing feature returns 422."""
    bad_record = dict(REPRESENTATIVE_VALID_RECORD)
    del bad_record["distance"]

    app = create_app(predictor=ModelPredictor(), metrics=FlightDelayMetrics())
    with TestClient(app) as client:
        resp = client.post("/api/v1/predictions", json=bad_record)
        assert resp.status_code == 422


def test_predict_single_out_of_bounds_rejected() -> None:
    """Test POST /api/v1/predictions rejects out of bounds values."""
    app = create_app(predictor=ModelPredictor(), metrics=FlightDelayMetrics())
    with TestClient(app) as client:
        # Invalid month 13
        rec1 = dict(REPRESENTATIVE_VALID_RECORD, month=13)
        assert client.post("/api/v1/predictions", json=rec1).status_code == 422

        # Invalid crs_dep_hour 24
        rec2 = dict(REPRESENTATIVE_VALID_RECORD, crs_dep_hour=24)
        assert client.post("/api/v1/predictions", json=rec2).status_code == 422

        # Invalid distance 0
        rec3 = dict(REPRESENTATIVE_VALID_RECORD, distance=0.0)
        assert client.post("/api/v1/predictions", json=rec3).status_code == 422

        # Invalid negative elapsed time
        rec4 = dict(REPRESENTATIVE_VALID_RECORD, crs_elapsed_time=-50.0)
        assert client.post("/api/v1/predictions", json=rec4).status_code == 422


def test_predict_single_target_leakage_rejected() -> None:
    """Test POST /api/v1/predictions strictly rejects post-departure target leakage."""
    app = create_app(predictor=ModelPredictor(), metrics=FlightDelayMetrics())
    with TestClient(app) as client:
        for leakage_field in [
            "arr_delay",
            "dep_delay",
            "actual_elapsed_time",
            "air_time",
            "wheels_off",
        ]:
            rec = dict(REPRESENTATIVE_VALID_RECORD)
            rec[leakage_field] = 25.0
            resp = client.post("/api/v1/predictions", json=rec)
            assert resp.status_code == 422, f"Expected 422 for leakage field '{leakage_field}'"


def test_predict_single_blank_categorical_rejected() -> None:
    """Test POST /api/v1/predictions rejects empty or blank categorical features."""
    app = create_app(predictor=ModelPredictor(), metrics=FlightDelayMetrics())
    with TestClient(app) as client:
        rec1 = dict(REPRESENTATIVE_VALID_RECORD, carrier="   ")
        assert client.post("/api/v1/predictions", json=rec1).status_code == 422

        rec2 = dict(REPRESENTATIVE_VALID_RECORD, origin="12")
        assert client.post("/api/v1/predictions", json=rec2).status_code == 422


def test_predict_batch_valid() -> None:
    """Test POST /api/v1/predictions/batch with valid records list."""
    predictor = ModelPredictor()
    app = create_app(predictor=predictor, metrics=FlightDelayMetrics())

    records = [
        dict(REPRESENTATIVE_VALID_RECORD, event_id=f"evt_{i}", distance=500.0 + i * 100)
        for i in range(5)
    ]

    with TestClient(app) as client:
        resp = client.post("/api/v1/predictions/batch", json={"records": records})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 5
        assert len(data["predictions"]) == 5
        assert data["predictions"][0]["event_id"] == "evt_0"


def test_predict_batch_exceeds_max_size_rejected() -> None:
    """Test POST /api/v1/predictions/batch rejects batch larger than max_batch_size."""
    settings = Settings(api_max_batch_size=3)
    app = create_app(settings=settings, predictor=ModelPredictor(), metrics=FlightDelayMetrics())

    records = [dict(REPRESENTATIVE_VALID_RECORD) for _ in range(4)]
    with TestClient(app) as client:
        resp = client.post("/api/v1/predictions/batch", json={"records": records})
        assert resp.status_code == 422
        assert "exceeds maximum allowed batch size" in resp.json()["detail"]


def test_predict_batch_empty_rejected() -> None:
    """Test POST /api/v1/predictions/batch rejects empty list."""
    app = create_app(predictor=ModelPredictor(), metrics=FlightDelayMetrics())
    with TestClient(app) as client:
        resp = client.post("/api/v1/predictions/batch", json={"records": []})
        assert resp.status_code == 422


def test_get_prediction_by_event_id_success() -> None:
    """Test GET /api/v1/predictions/{event_id} when record exists in storage."""
    mock_storage = MagicMock()
    mock_storage.is_connected = True

    now = datetime.datetime.now(datetime.UTC)
    mock_record = PredictionRecord(
        event_id="abc12345",
        carrier="AA",
        flight_number="100",
        origin="JFK",
        dest="LAX",
        event_time=now,
        prediction_time=now,
        prediction_date=now.date(),
        delay_probability=0.35,
        predicted_delayed=False,
        model_version="1.0.0",
    )
    mock_storage.get_prediction_by_event.return_value = mock_record

    app = create_app(storage=mock_storage, metrics=FlightDelayMetrics())
    with TestClient(app) as client:
        resp = client.get("/api/v1/predictions/abc12345")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_id"] == "abc12345"
        assert data["carrier"] == "AA"
        assert data["delay_probability"] == 0.35
        assert data["predicted_delayed"] is False


def test_get_prediction_by_event_id_not_found() -> None:
    """Test GET /api/v1/predictions/{event_id} returns 404 when not found."""
    mock_storage = MagicMock()
    mock_storage.is_connected = True
    mock_storage.get_prediction_by_event.return_value = None

    app = create_app(storage=mock_storage, metrics=FlightDelayMetrics())
    with TestClient(app) as client:
        resp = client.get("/api/v1/predictions/nonexistent_id")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]


def test_get_prediction_by_event_id_storage_unavailable() -> None:
    """Test GET /api/v1/predictions/{event_id} returns 503 when Cassandra is disconnected."""
    mock_storage = MagicMock()
    mock_storage.is_connected = False

    app = create_app(storage=mock_storage, metrics=FlightDelayMetrics())
    with TestClient(app) as client:
        resp = client.get("/api/v1/predictions/test_event")
        assert resp.status_code == 503
