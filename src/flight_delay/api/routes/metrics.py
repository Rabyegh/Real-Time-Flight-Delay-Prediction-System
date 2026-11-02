"""Metrics endpoint exposing Prometheus text format."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from flight_delay.api.dependencies import get_metrics
from flight_delay.monitoring.metrics import FlightDelayMetrics

router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
def metrics(metrics_mgr: Annotated[FlightDelayMetrics, Depends(get_metrics)]) -> Response:
    """Expose Prometheus metrics for scraping."""
    content, media_type = metrics_mgr.generate_metrics_text()
    return Response(content=content, media_type=media_type)
