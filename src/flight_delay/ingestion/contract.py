"""Kafka event schema, validation, and contract definitions for real-time flight ingestion.

Timestamp Semantics:
- `crs_dep_time`, `crs_dep_hour`, `crs_dep_minute`, `crs_arr_hour`, and `crs_arr_minute`
  represent local airport scheduled departure/arrival times, conforming to the historical
  BTS standard and matching the Phase 2/3 feature definition.
- `event_timestamp` and `ingestion_timestamp` represent streaming event occurrence and ingestion
  times in UTC (ISO 8601 with trailing 'Z' or '+00:00'), which are used for Spark Structured
  Streaming watermarking and stateful deduplication.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)

SCHEMA_VERSION = "1.0.0"

# Target and actual post-departure outcome fields prohibited in pre-departure streaming features
PROHIBITED_LEAKAGE_COLUMNS = frozenset(
    {
        "dep_delay",
        "arr_delay",
        "dep_time",
        "arr_time",
        "is_delayed_15",
        "actual_elapsed_time",
        "air_time",
        "taxi_in",
        "taxi_out",
        "wheels_off",
        "wheels_on",
        "cancellation_code",
        "cancelled",
        "diverted",
    }
)

IATA_CODE_REGEX = re.compile(r"^[A-Z0-9]{3}$")
CARRIER_CODE_REGEX = re.compile(r"^[A-Z0-9]{2,3}$")
TIME_BLK_REGEX = re.compile(r"^\d{4}-\d{4}$")


class ContractValidationError(ValueError):
    """Raised when an event violates schema, types, bounds, or leakage rules."""


@dataclass(frozen=True)
class FlightEvent:
    """Validated pre-departure flight event payload."""

    event_id: str
    schema_version: str
    event_timestamp: str
    ingestion_timestamp: str
    source: dict[str, Any]
    carrier: str
    flight_number: str
    origin: str
    dest: str
    fl_date: str
    crs_dep_time: str
    crs_dep_hour: int
    crs_dep_minute: int
    dep_time_blk: str
    crs_arr_hour: int
    crs_arr_minute: int
    arr_time_blk: str
    month: int
    day_of_month: int
    day_of_week: int
    is_weekend: int
    crs_elapsed_time: float
    distance: float

    def to_dict(self) -> dict[str, Any]:
        """Convert flight event to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize flight event to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True)
class DeadLetterRecord:
    """Record container for rejected or malformed events routed to dead-letter storage."""

    raw_payload: str
    rejection_reason: str
    error_code: str
    rejection_timestamp: str
    event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert dead letter record to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dead letter record to JSON string."""
        return json.dumps(self.to_dict())


def generate_event_id(
    carrier: str,
    flight_number: str,
    origin: str,
    dest: str,
    fl_date: str,
    crs_dep_time: str,
) -> str:
    """Generate a deterministic SHA-256 event ID for reproducible identity and deduplication."""
    composite_key = (
        f"{carrier.strip().upper()}:"
        f"{flight_number.strip().upper()}:"
        f"{origin.strip().upper()}:"
        f"{dest.strip().upper()}:"
        f"{fl_date.strip()}:"
        f"{crs_dep_time.strip()}"
    )
    return hashlib.sha256(composite_key.encode("utf-8")).hexdigest()[:32]


def extract_dict_keys_recursive(data: Any) -> set[str]:
    """Recursively extract all dictionary keys in lowercase from a nested structure."""
    keys: set[str] = set()
    stack = [data]
    while stack:
        curr = stack.pop()
        if isinstance(curr, dict):
            for k, v in curr.items():
                keys.add(str(k).strip().lower())
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(curr, list):
            for item in curr:
                if isinstance(item, (dict, list)):
                    stack.append(item)
    return keys


def contains_prohibited_leakage(payload: dict[str, Any] | str | None) -> bool:
    """Check whether a dict or raw JSON string contains prohibited leakage keys."""
    if payload is None:
        return False
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except Exception:
            return False
    elif isinstance(payload, dict):
        data = payload
    else:
        return False

    keys = extract_dict_keys_recursive(data)
    return bool(keys.intersection(PROHIBITED_LEAKAGE_COLUMNS))


def validate_flight_event_dict(payload: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    """Validate a raw flight event dictionary against the Phase 5 event contract.

    Returns:
        tuple[is_valid, error_code, rejection_reason]
    """
    if not isinstance(payload, dict):
        return False, "INVALID_TYPE", f"Payload must be a dict, got {type(payload).__name__}"

    # 1. Check for prohibited leakage columns (recursively across all keys)
    keys_found = extract_dict_keys_recursive(payload)
    leakage_found = keys_found.intersection(PROHIBITED_LEAKAGE_COLUMNS)
    if leakage_found:
        return (
            False,
            "LEAKAGE_DETECTED",
            f"Prohibited post-departure leakage fields present: {sorted(leakage_found)}",
        )

    # 2. Check required envelope fields
    required_envelope = {
        "event_id",
        "schema_version",
        "event_timestamp",
        "ingestion_timestamp",
        "source",
    }
    missing_envelope = required_envelope - set(payload.keys())
    if missing_envelope:
        return False, "MISSING_ENVELOPE", f"Missing envelope fields: {sorted(missing_envelope)}"

    # Check that event_id is not null, empty, or whitespace-only
    event_id_val = payload.get("event_id")
    if not isinstance(event_id_val, str) or not event_id_val.strip():
        return (
            False,
            "INVALID_EVENT_ID",
            "event_id cannot be null, empty, or whitespace-only",
        )

    # 3. Check schema version
    version = str(payload.get("schema_version", ""))
    if version != SCHEMA_VERSION:
        return (
            False,
            "UNSUPPORTED_VERSION",
            f"Expected schema_version '{SCHEMA_VERSION}', got '{version}'",
        )

    # 4. Check timestamps
    for ts_field in ("event_timestamp", "ingestion_timestamp"):
        ts_val = payload.get(ts_field)
        if not isinstance(ts_val, str):
            return (
                False,
                "INVALID_TIMESTAMP",
                f"{ts_field} must be a string, got {type(ts_val).__name__}",
            )
        try:
            # Parse ISO 8601
            datetime.datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        except (ValueError, TypeError) as exc:
            return (
                False,
                "INVALID_TIMESTAMP",
                f"{ts_field} '{ts_val}' is not valid ISO 8601: {exc}",
            )

    # 5. Check required flight fields
    required_flight_fields = {
        "carrier",
        "flight_number",
        "origin",
        "dest",
        "fl_date",
        "crs_dep_time",
        "crs_dep_hour",
        "crs_dep_minute",
        "dep_time_blk",
        "crs_arr_hour",
        "crs_arr_minute",
        "arr_time_blk",
        "month",
        "day_of_month",
        "day_of_week",
        "is_weekend",
        "crs_elapsed_time",
        "distance",
    }
    missing_flight = required_flight_fields - set(payload.keys())
    if missing_flight:
        return False, "MISSING_FIELDS", f"Missing required flight fields: {sorted(missing_flight)}"

    # 6. Validate types and ranges
    try:
        carrier = str(payload["carrier"]).strip().upper()
        if not CARRIER_CODE_REGEX.match(carrier):
            return False, "INVALID_CARRIER", f"Invalid carrier code: {carrier}"

        origin = str(payload["origin"]).strip().upper()
        dest = str(payload["dest"]).strip().upper()
        if not IATA_CODE_REGEX.match(origin):
            return False, "INVALID_ORIGIN", f"Invalid origin IATA code: {origin}"
        if not IATA_CODE_REGEX.match(dest):
            return False, "INVALID_DEST", f"Invalid destination IATA code: {dest}"
        if origin == dest:
            return False, "EQUAL_ORIGIN_DEST", f"Origin and destination cannot be equal: {origin}"

        # Date validation
        fl_date_str = str(payload["fl_date"]).strip()
        datetime.date.fromisoformat(fl_date_str)

        # Clocks and calendar
        dep_hour = int(payload["crs_dep_hour"])
        dep_min = int(payload["crs_dep_minute"])
        arr_hour = int(payload["crs_arr_hour"])
        arr_min = int(payload["crs_arr_minute"])
        month = int(payload["month"])
        day = int(payload["day_of_month"])
        dow = int(payload["day_of_week"])
        weekend = int(payload["is_weekend"])

        if not (0 <= dep_hour <= 23 and 0 <= dep_min <= 59):
            return (
                False,
                "INVALID_DEP_TIME",
                f"Invalid departure clock: hour={dep_hour}, min={dep_min}",
            )
        if not (0 <= arr_hour <= 23 and 0 <= arr_min <= 59):
            return (
                False,
                "INVALID_ARR_TIME",
                f"Invalid arrival clock: hour={arr_hour}, min={arr_min}",
            )
        if not (1 <= month <= 12 and 1 <= day <= 31 and 1 <= dow <= 7):
            return False, "INVALID_CALENDAR", f"Invalid calendar: M={month}, D={day}, DOW={dow}"
        if weekend not in (0, 1):
            return False, "INVALID_WEEKEND", f"is_weekend must be 0 or 1, got {weekend}"

        dep_blk = str(payload["dep_time_blk"]).strip()
        arr_blk = str(payload["arr_time_blk"]).strip()
        if not TIME_BLK_REGEX.match(dep_blk):
            return False, "INVALID_TIME_BLOCK", f"Invalid dep_time_blk: {dep_blk}"
        if not TIME_BLK_REGEX.match(arr_blk):
            return False, "INVALID_TIME_BLOCK", f"Invalid arr_time_blk: {arr_blk}"

        elapsed = float(payload["crs_elapsed_time"])
        distance = float(payload["distance"])
        if elapsed <= 0.0:
            return False, "INVALID_ELAPSED_TIME", f"crs_elapsed_time must be positive: {elapsed}"
        if distance <= 0.0:
            return False, "INVALID_DISTANCE", f"distance must be positive: {distance}"

    except (ValueError, TypeError) as exc:
        return False, "TYPE_OR_VALUE_ERROR", f"Field parsing/conversion error: {exc}"

    return True, None, None


def get_spark_flight_event_schema() -> StructType:
    """Return explicit PySpark StructType schema for parsing flight event JSON payloads."""
    return StructType(
        [
            StructField("_corrupt_record", StringType(), True),
            StructField("event_id", StringType(), True),
            StructField("schema_version", StringType(), True),
            StructField("event_timestamp", StringType(), True),
            StructField("ingestion_timestamp", StringType(), True),
            StructField("source", MapType(StringType(), StringType()), True),
            StructField("carrier", StringType(), True),
            StructField("flight_number", StringType(), True),
            StructField("origin", StringType(), True),
            StructField("dest", StringType(), True),
            StructField("fl_date", StringType(), True),
            StructField("crs_dep_time", StringType(), True),
            StructField("crs_dep_hour", IntegerType(), True),
            StructField("crs_dep_minute", IntegerType(), True),
            StructField("dep_time_blk", StringType(), True),
            StructField("crs_arr_hour", IntegerType(), True),
            StructField("crs_arr_minute", IntegerType(), True),
            StructField("arr_time_blk", StringType(), True),
            StructField("month", IntegerType(), True),
            StructField("day_of_month", IntegerType(), True),
            StructField("day_of_week", IntegerType(), True),
            StructField("is_weekend", IntegerType(), True),
            StructField("crs_elapsed_time", DoubleType(), True),
            StructField("distance", DoubleType(), True),
        ]
    )


def get_spark_dead_letter_schema() -> StructType:
    """Return explicit PySpark StructType schema for dead-letter records."""
    return StructType(
        [
            StructField("event_id", StringType(), True),
            StructField("raw_payload", StringType(), True),
            StructField("error_code", StringType(), True),
            StructField("rejection_reason", StringType(), True),
            StructField("rejection_timestamp", StringType(), True),
        ]
    )
