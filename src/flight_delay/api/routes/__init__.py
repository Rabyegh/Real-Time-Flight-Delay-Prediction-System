"""API route subpackages."""

from flight_delay.api.routes.airports import router as airports_router
from flight_delay.api.routes.health import router as health_router
from flight_delay.api.routes.metrics import router as metrics_router
from flight_delay.api.routes.predictions import router as predictions_router

__all__ = [
    "airports_router",
    "health_router",
    "metrics_router",
    "predictions_router",
]
