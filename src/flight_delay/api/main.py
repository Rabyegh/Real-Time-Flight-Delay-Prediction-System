"""FastAPI application entry point for the Flight Delay Prediction service."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from flight_delay.api.middleware import MetricsMiddleware
from flight_delay.api.routes import (
    airports_router,
    health_router,
    metrics_router,
    predictions_router,
)
from flight_delay.config.settings import Settings, get_settings
from flight_delay.monitoring.metrics import FlightDelayMetrics
from flight_delay.storage.cassandra import (
    CassandraStorage,
    StorageConnectionError,
    StorageQueryError,
    StorageValidationError,
)
from flight_delay.training.inference import (
    InferenceError,
    ModelPredictor,
    SchemaValidationError,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings

    # 1. Initialize metrics manager if not already injected
    if not hasattr(app.state, "metrics") or app.state.metrics is None:
        app.state.metrics = FlightDelayMetrics()

    # 2. Initialize ModelPredictor if not already injected
    if not hasattr(app.state, "predictor") or app.state.predictor is None:
        try:
            predictor = ModelPredictor(
                model_path=settings.model_artifact_path,
                threshold=settings.prediction_threshold,
                model_version=settings.app_version,
            )
            app.state.predictor = predictor
            logger.info("Loaded model artifact from %s", settings.model_artifact_path)
        except Exception as exc:
            logger.error("Failed to load model artifact at startup: %s", exc)
            app.state.predictor = None

    # 3. Initialize CassandraStorage if not already injected
    if not hasattr(app.state, "storage") or app.state.storage is None:
        try:
            storage = CassandraStorage(
                contact_points=[settings.cassandra_host],
                port=settings.cassandra_port,
                keyspace=settings.cassandra_keyspace,
            )
            try:
                storage.connect()
                app.state.metrics.set_cassandra_status(True)
                logger.info(
                    "Connected to Cassandra at %s:%d/%s",
                    settings.cassandra_host,
                    settings.cassandra_port,
                    settings.cassandra_keyspace,
                )
            except Exception as exc:
                logger.warning(
                    "Cassandra cluster unreachable at startup (%s). Operating in degraded mode.",
                    exc,
                )
                app.state.metrics.set_cassandra_status(False)
            app.state.storage = storage
        except Exception as exc:
            logger.error("Failed to instantiate CassandraStorage: %s", exc)
            app.state.storage = None
            app.state.metrics.set_cassandra_status(False)

    yield

    # Clean shutdown
    storage = getattr(app.state, "storage", None)
    if storage is not None and bool(storage.is_connected):
        try:
            storage.close()
            logger.info("Closed Cassandra storage connection.")
        except Exception as exc:
            logger.warning("Error closing Cassandra connection: %s", exc)


def create_app(
    settings: Settings | None = None,
    predictor: ModelPredictor | None = None,
    storage: CassandraStorage | None = None,
    metrics: FlightDelayMetrics | None = None,
) -> FastAPI:
    """Application factory for the flight delay prediction REST API."""
    active_settings = settings or get_settings()

    app = FastAPI(
        title="Flight Delay Prediction API",
        description=(
            "Production REST API serving pre-departure flight delay predictions "
            "and Cassandra queries."
        ),
        version=active_settings.app_version,
        lifespan=lifespan,
    )

    # Attach injected instances to application state if provided
    app.state.settings = active_settings
    if predictor is not None:
        app.state.predictor = predictor
    if storage is not None:
        app.state.storage = storage
    if metrics is not None:
        app.state.metrics = metrics

    # Custom exception handlers for clean HTTP status codes
    @app.exception_handler(SchemaValidationError)
    async def schema_validation_handler(
        request: Request, exc: SchemaValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "error_type": "SchemaValidationError"},
        )

    @app.exception_handler(StorageValidationError)
    async def storage_validation_handler(
        request: Request, exc: StorageValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "error_type": "StorageValidationError"},
        )

    @app.exception_handler(StorageConnectionError)
    async def storage_connection_handler(
        request: Request, exc: StorageConnectionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "Cassandra storage connection failed.",
                "error_type": "StorageConnectionError",
            },
        )

    @app.exception_handler(StorageQueryError)
    async def storage_query_handler(request: Request, exc: StorageQueryError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "Cassandra query failed or timed out.",
                "error_type": "StorageQueryError",
            },
        )

    @app.exception_handler(InferenceError)
    async def inference_error_handler(request: Request, exc: InferenceError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "error_type": "InferenceError"},
        )

    # Attach timing and metrics middleware
    app.add_middleware(MetricsMiddleware)

    # Register routers
    app.include_router(health_router)
    app.include_router(predictions_router)
    app.include_router(airports_router)
    app.include_router(metrics_router)

    return app


# Module-level default application instance
app = create_app()
