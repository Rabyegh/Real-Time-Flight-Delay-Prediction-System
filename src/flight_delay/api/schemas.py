"""Pydantic schemas for the FastAPI prediction service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flight_delay.training.inference import ALL_PROHIBITED_LEAKAGE_FIELDS


class FlightFeatures(BaseModel):
    """Strict validation schema for model input features.

    Matches the authoritative 15-feature contract required by the trained model pipeline.
    Explicitly rejects post-departure leakage columns.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Categorical features
    carrier: str = Field(..., min_length=1, max_length=10, description="Carrier code (e.g. AA, DL)")
    origin: str = Field(
        ..., min_length=3, max_length=3, description="3-letter origin airport IATA code"
    )
    dest: str = Field(
        ..., min_length=3, max_length=3, description="3-letter destination airport IATA code"
    )
    dep_time_blk: str = Field(
        ..., min_length=1, max_length=10, description="Departure time block (e.g. 0600-0659)"
    )
    arr_time_blk: str = Field(
        ..., min_length=1, max_length=10, description="Arrival time block (e.g. 0800-0859)"
    )

    # Numerical time & calendar features
    crs_dep_hour: int = Field(..., ge=0, le=23, description="Scheduled departure hour (0-23)")
    crs_dep_minute: int = Field(..., ge=0, le=59, description="Scheduled departure minute (0-59)")
    crs_arr_hour: int = Field(..., ge=0, le=23, description="Scheduled arrival hour (0-23)")
    crs_arr_minute: int = Field(..., ge=0, le=59, description="Scheduled arrival minute (0-59)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    day_of_week: int = Field(..., ge=1, le=7, description="Day of week (1=Monday, 7=Sunday)")
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend flag (0 or 1)")

    # Flight duration & distance
    crs_elapsed_time: float = Field(
        ..., gt=0.0, description="Scheduled elapsed time in minutes (> 0)"
    )
    distance: float = Field(..., gt=0.0, description="Flight distance in miles (> 0)")

    # Optional metadata identifiers (preserved, not passed to model)
    event_id: str | None = Field(
        default=None, description="Optional deterministic event identifier"
    )
    flight_number: str | None = Field(default=None, description="Optional flight number")
    event_time: str | None = Field(default=None, description="Optional event timestamp")
    prediction_time: str | None = Field(
        default=None, description="Optional fixed prediction timestamp"
    )

    @field_validator("carrier", "origin", "dest", "dep_time_blk", "arr_time_blk")
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        s = v.strip()
        if not s or s.lower() == "none" or s.lower() == "nan" or s.lower() == "null":
            raise ValueError("Categorical feature must not be empty or null-like.")
        return s

    @field_validator("origin", "dest")
    @classmethod
    def validate_airport_code(cls, v: str) -> str:
        s = v.strip().upper()
        if len(s) != 3 or not s.isalpha():
            raise ValueError(f"Airport code must be 3 alphabetic characters, got '{v}'")
        return s

    @model_validator(mode="before")
    @classmethod
    def check_for_target_leakage(cls, values: Any) -> Any:
        if isinstance(values, dict):
            keys = set(values.keys())
            lower_to_orig = {k.lower(): k for k in keys}
            leakage_present = []
            for leakage_col in ALL_PROHIBITED_LEAKAGE_FIELDS:
                if leakage_col.lower() in lower_to_orig:
                    orig_key = lower_to_orig[leakage_col.lower()]
                    val = values[orig_key]
                    if val is not None and str(val).strip().lower() not in (
                        "",
                        "none",
                        "nan",
                        "null",
                    ):
                        leakage_present.append(orig_key)
            if leakage_present:
                raise ValueError(
                    f"Post-departure target leakage columns detected: {leakage_present}. "
                    "Pre-departure prediction rejects actual flight observations."
                )
        return values

    def to_model_dict(self) -> dict[str, Any]:
        """Convert to dictionary containing model features and optional metadata."""
        d = self.model_dump()
        return d


class PredictionResponse(BaseModel):
    """Response schema for a single delay prediction."""

    delay_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Model probability of delay >= 15 min"
    )
    predicted_delayed: bool = Field(
        ..., description="Binary classification (True if delay_probability >= threshold)"
    )
    is_delayed: bool = Field(..., description="Alias for predicted_delayed")
    threshold: float = Field(..., description="Classification threshold applied")
    model_version: str = Field(..., description="Model version identifier")
    prediction_time: str = Field(..., description="UTC timestamp when prediction was computed")
    event_id: str | None = Field(default=None, description="Event ID if supplied or generated")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""

    records: list[FlightFeatures] = Field(..., min_length=1, description="List of flight records")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    predictions: list[PredictionResponse] = Field(..., description="List of prediction results")
    count: int = Field(..., ge=0, description="Number of predictions returned")
    model_version: str = Field(..., description="Model version identifier")


class HealthResponse(BaseModel):
    """Response schema for system health endpoint."""

    status: str = Field(..., description="Overall service status (ok, degraded, unhealthy)")
    version: str = Field(..., description="Application version")
    model_ready: bool = Field(..., description="Whether inference model is loaded and ready")
    cassandra_connected: bool = Field(
        ..., description="Whether Cassandra storage connection is active"
    )


class PersistedPredictionResponse(BaseModel):
    """Response schema for a persisted prediction record retrieved from Cassandra."""

    event_id: str
    carrier: str
    flight_number: str
    origin: str
    dest: str
    event_time: str
    prediction_time: str
    prediction_date: str
    delay_probability: float
    predicted_delayed: bool
    model_version: str


class AirportPredictionsResponse(BaseModel):
    """Response schema for airport predictions query."""

    airport: str
    date: str
    limit: int
    count: int
    predictions: list[PersistedPredictionResponse]


class ErrorResponse(BaseModel):
    """Standard structured error response."""

    detail: str
    error_type: str | None = None
