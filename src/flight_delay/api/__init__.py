"""HTTP API package for Flight Delay Prediction service."""

from flight_delay.api.main import app, create_app

__all__ = ["app", "create_app"]
