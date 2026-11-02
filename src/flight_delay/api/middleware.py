"""HTTP middleware for request timing, error tracking, and metrics collection."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

from flight_delay.monitoring.metrics import FlightDelayMetrics

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware that records API request counts, latencies, and server errors."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        metrics: FlightDelayMetrics | None = getattr(request.app.state, "metrics", None)
        start_time = time.perf_counter()
        endpoint = request.url.path
        method = request.method

        try:
            response = await call_next(request)
            duration = time.perf_counter() - start_time
            if metrics is not None:
                # Collapse dynamic path params if route is matched
                matched_route = None
                for route in request.app.routes:
                    match, _ = route.matches(request.scope)
                    if match == Match.FULL:
                        matched_route = getattr(route, "path", None)
                        break
                path_label = matched_route or endpoint

                metrics.record_api_request(
                    endpoint=path_label,
                    method=method,
                    status_code=response.status_code,
                    duration_seconds=duration,
                )
                if response.status_code >= 500:
                    metrics.record_api_error(
                        endpoint=path_label, error_type=f"HTTP_{response.status_code}"
                    )
            return response

        except Exception as exc:
            duration = time.perf_counter() - start_time
            if metrics is not None:
                metrics.record_api_request(
                    endpoint=endpoint,
                    method=method,
                    status_code=500,
                    duration_seconds=duration,
                )
                metrics.record_api_error(endpoint=endpoint, error_type=type(exc).__name__)
            raise
