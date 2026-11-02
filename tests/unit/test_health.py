"""Tests for the API health endpoint."""

from fastapi.testclient import TestClient

from flight_delay.api.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_ready" in data
        assert "cassandra_connected" in data
        assert data["version"] == "0.1.0"
