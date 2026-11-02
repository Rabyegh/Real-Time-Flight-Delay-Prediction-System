"""Dependency injection providers for FastAPI endpoints."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status

from flight_delay.config.settings import Settings, get_settings
from flight_delay.monitoring.metrics import FlightDelayMetrics
from flight_delay.storage.cassandra import CassandraStorage
from flight_delay.training.inference import ModelPredictor

logger = logging.getLogger(__name__)


def get_settings_dep(request: Request) -> Settings:
    """Provide application configuration settings."""
    return getattr(request.app.state, "settings", None) or get_settings()


def get_predictor(request: Request) -> ModelPredictor:
    """Retrieve the loaded ModelPredictor singleton from application state.

    Raises 503 Service Unavailable if the model failed to initialize.
    """
    predictor: ModelPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not ready or failed to load.",
        )
    return predictor


def get_optional_predictor(request: Request) -> ModelPredictor | None:
    """Retrieve ModelPredictor if available, without raising an exception."""
    return getattr(request.app.state, "predictor", None)


def get_storage(request: Request) -> CassandraStorage:
    """Retrieve an active CassandraStorage instance from application state.

    Raises 503 Service Unavailable if Cassandra is offline or uninitialized.
    """
    storage: CassandraStorage | None = getattr(request.app.state, "storage", None)
    if storage is None or not bool(storage.is_connected):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cassandra storage service is currently unavailable.",
        )
    return storage


def get_optional_storage(request: Request) -> CassandraStorage | None:
    """Retrieve CassandraStorage instance if present, without requiring connectivity."""
    return getattr(request.app.state, "storage", None)


def get_metrics(request: Request) -> FlightDelayMetrics:
    """Retrieve the metrics manager from application state."""
    metrics: FlightDelayMetrics | None = getattr(request.app.state, "metrics", None)
    if metrics is None:
        # Fallback to an isolated instance if state not populated
        metrics = FlightDelayMetrics()
        request.app.state.metrics = metrics
    return metrics
