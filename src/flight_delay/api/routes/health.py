"""System health route."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from flight_delay.api.dependencies import (
    get_optional_predictor,
    get_optional_storage,
    get_settings_dep,
)
from flight_delay.api.schemas import HealthResponse
from flight_delay.config.settings import Settings
from flight_delay.storage.cassandra import CassandraStorage
from flight_delay.training.inference import ModelPredictor

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    request: Request,
    predictor: Annotated[ModelPredictor | None, Depends(get_optional_predictor)],
    storage: Annotated[CassandraStorage | None, Depends(get_optional_storage)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> HealthResponse:
    """Report system health, model readiness, and Cassandra connectivity.

    Reports Cassandra status independently from overall API availability without
    exposing credentials, keyspace details, or internal host names.
    """
    model_ready = predictor is not None and getattr(predictor, "_pipeline", None) is not None
    cassandra_connected = bool(storage is not None and storage.is_connected)

    if not model_ready:
        status_label = "unhealthy"
    elif not cassandra_connected:
        status_label = "degraded"
    else:
        status_label = "ok"

    return HealthResponse(
        status=status_label,
        version=settings.app_version,
        model_ready=model_ready,
        cassandra_connected=cassandra_connected,
    )
