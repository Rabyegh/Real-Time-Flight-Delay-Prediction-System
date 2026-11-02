"""Airport predictions query endpoint."""

from __future__ import annotations

import datetime
import logging
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from flight_delay.api.dependencies import get_storage
from flight_delay.api.schemas import (
    AirportPredictionsResponse,
    PersistedPredictionResponse,
)
from flight_delay.storage.cassandra import (
    CassandraStorage,
    PredictionRecord,
    StorageValidationError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/airports", tags=["airports"])
IATA_REGEX = re.compile(r"^[A-Z]{3}$")


@router.get("/{airport}/predictions", response_model=AirportPredictionsResponse)
def get_airport_predictions(
    airport: Annotated[str, Path(description="3-letter airport IATA code (e.g. ORD, ATL)")],
    storage: Annotated[CassandraStorage, Depends(get_storage)],
    date: Annotated[
        str | None,
        Query(
            description="Query calendar date bucket in YYYY-MM-DD format (defaults to UTC date)",
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=1000,
            description="Maximum number of predictions to retrieve (1 to 1000)",
        ),
    ] = 100,
) -> AirportPredictionsResponse:
    """Retrieve recent flight delay predictions for an airport within a specific date bucket."""
    airport_code = airport.strip().upper()
    if not IATA_REGEX.match(airport_code):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid airport code '{airport}'. Must be a 3-letter alphabetic IATA code.",
        )

    bucket_date: datetime.date
    if date is None or not date.strip():
        bucket_date = datetime.datetime.now(datetime.UTC).date()
    else:
        try:
            bucket_date = datetime.date.fromisoformat(date.strip())
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid date format '{date}'. Expected YYYY-MM-DD.",
            ) from exc

    try:
        records: list[PredictionRecord] = storage.get_predictions_by_airport(
            airport=airport_code,
            date=bucket_date,
            limit=limit,
        )
    except StorageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    responses: list[PersistedPredictionResponse] = []
    for rec in records:
        event_time_str = (
            rec.event_time.isoformat()
            if hasattr(rec.event_time, "isoformat")
            else str(rec.event_time)
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
        responses.append(
            PersistedPredictionResponse(
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
        )

    return AirportPredictionsResponse(
        airport=airport_code,
        date=bucket_date.isoformat(),
        limit=limit,
        count=len(responses),
        predictions=responses,
    )
