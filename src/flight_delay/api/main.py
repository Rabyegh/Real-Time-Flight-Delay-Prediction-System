"""FastAPI application entry point."""

from fastapi import FastAPI

from flight_delay.api.routes.health import router as health_router

app = FastAPI(title="Flight Delay Prediction API", version="0.1.0")
app.include_router(health_router)
