"""Prediction endpoints for single, batch, and event-retrieval operations."""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from flight_delay.api.dependencies import (
    get_metrics,
    get_predictor,
    get_settings_dep,
    get_storage,
)
from flight_delay.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FlightFeatures,
    PersistedPredictionResponse,
    PredictionResponse,
)
from flight_delay.config.settings import Settings
from flight_delay.monitoring.metrics import FlightDelayMetrics
from flight_delay.storage.cassandra import CassandraStorage, PredictionRecord
from flight_delay.training.inference import ModelPredictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_single(
    flight: FlightFeatures,
    predictor: Annotated[ModelPredictor, Depends(get_predictor)],
    metrics: Annotated[FlightDelayMetrics, Depends(get_metrics)],
) -> PredictionResponse:
    """Evaluate flight delay prediction for a single flight record."""
    start_time = time.perf_counter()
    try:
        rec_dict = flight.to_model_dict()
        res = predictor.predict_single(rec_dict)
        duration = time.perf_counter() - start_time

        is_delayed_bool = bool(res.predicted_delayed)
        metrics.record_prediction(
            model_version=res.model_version,
            is_delayed=is_delayed_bool,
            duration_seconds=duration,
            threshold=predictor.threshold,
        )

        return PredictionResponse(
            delay_probability=res.delay_probability,
            predicted_delayed=is_delayed_bool,
            is_delayed=is_delayed_bool,
            threshold=predictor.threshold,
            model_version=res.model_version,
            prediction_time=res.prediction_time,
            event_id=res.event_id or flight.event_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        metrics.record_prediction_error(error_type=type(exc).__name__)
        raise


@router.post("/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_batch(
    batch: BatchPredictionRequest,
    predictor: Annotated[ModelPredictor, Depends(get_predictor)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    metrics: Annotated[FlightDelayMetrics, Depends(get_metrics)],
) -> BatchPredictionResponse:
    """Evaluate flight delay predictions for a bounded batch of flight records."""
    if len(batch.records) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch prediction request cannot be empty.",
        )

    if len(batch.records) > settings.api_max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Batch size {len(batch.records)} exceeds maximum allowed batch size "
                f"of {settings.api_max_batch_size}."
            ),
        )

    start_time = time.perf_counter()
    try:
        records_data = [f.to_model_dict() for f in batch.records]
        results = predictor.predict(records_data)
        duration = time.perf_counter() - start_time

        responses: list[PredictionResponse] = []
        for i, res in enumerate(results):
            is_delayed_bool = bool(res.predicted_delayed)
            # Record per-prediction metrics
            metrics.record_prediction(
                model_version=res.model_version,
                is_delayed=is_delayed_bool,
                duration_seconds=duration / max(1, len(results)),
                threshold=predictor.threshold,
            )
            responses.append(
                PredictionResponse(
                    delay_probability=res.delay_probability,
                    predicted_delayed=is_delayed_bool,
                    is_delayed=is_delayed_bool,
                    threshold=predictor.threshold,
                    model_version=res.model_version,
                    prediction_time=res.prediction_time,
                    event_id=res.event_id or batch.records[i].event_id,
                )
            )

        return BatchPredictionResponse(
            predictions=responses,
            count=len(responses),
            model_version=predictor.model_version,
        )
    except HTTPException:
        raise
    except Exception as exc:
        metrics.record_prediction_error(error_type=type(exc).__name__)
        raise


@router.get("/{event_id}", response_model=PersistedPredictionResponse)
def get_prediction_by_event(
    event_id: Annotated[
        str,
        Path(min_length=1, max_length=128, description="Deterministic event ID"),
    ],
    storage: Annotated[CassandraStorage, Depends(get_storage)],
) -> PersistedPredictionResponse:
    """Retrieve a persisted flight delay prediction by its deterministic event ID."""
    clean_id = event_id.strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="event_id must not be empty.",
        )

    rec: PredictionRecord | None = storage.get_prediction_by_event(clean_id)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction for event_id '{clean_id}' not found.",
        )

    event_time_str = (
        rec.event_time.isoformat() if hasattr(rec.event_time, "isoformat") else str(rec.event_time)
    )
    pred_time_str = (
        rec.prediction_time.isoformat()
        if hasattr(rec.prediction_time, "isoformat")
        else str(rec.prediction_time)
    )
    pred_date_str = (
        rec.prediction_date.isoformat()
        if hasattr(rec.prediction_date, "isoformat")
        else str(rec.prediction_date)
    )

    return PersistedPredictionResponse(
        event_id=rec.event_id,
        carrier=rec.carrier,
        flight_number=rec.flight_number,
        origin=rec.origin,
        dest=rec.dest,
        event_time=event_time_str,
        prediction_time=pred_time_str,
        prediction_date=pred_date_str,
        delay_probability=rec.delay_probability,
        predicted_delayed=rec.predicted_delayed,
        model_version=rec.model_version,
    )
