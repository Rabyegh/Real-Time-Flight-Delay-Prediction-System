"""Unit tests for Phase 5 Kafka event schema, validation, and contract definitions."""

import pytest
from pyspark.sql.types import StructType

from flight_delay.ingestion.contract import (
    PROHIBITED_LEAKAGE_COLUMNS,
    SCHEMA_VERSION,
    FlightEvent,
    contains_prohibited_leakage,
    generate_event_id,
    get_spark_dead_letter_schema,
    get_spark_flight_event_schema,
    validate_flight_event_dict,
)
from flight_delay.ingestion.kafka_producer import generate_synthetic_flight_event


def test_generate_event_id_is_deterministic() -> None:
    id1 = generate_event_id("AA", "AA100", "JFK", "LAX", "2023-06-25", "0830")
    id2 = generate_event_id("AA", "AA100", "JFK", "LAX", "2023-06-25", "0830")
    assert id1 == id2
    assert len(id1) == 32

    # Case insensitivity and whitespace stripping
    id3 = generate_event_id("aa ", " aa100", " jfk ", " lax", "2023-06-25", "0830")
    assert id1 == id3

    # Different flight yields different ID
    id_diff = generate_event_id("DL", "DL200", "ATL", "MIA", "2023-06-25", "0830")
    assert id1 != id_diff


def test_synthetic_event_passes_contract_validation() -> None:
    event = generate_synthetic_flight_event(seed=42)
    assert isinstance(event, FlightEvent)
    assert event.schema_version == SCHEMA_VERSION

    event_dict = event.to_dict()
    is_valid, error_code, reason = validate_flight_event_dict(event_dict)
    assert is_valid is True
    assert error_code is None
    assert reason is None


def test_validate_flight_event_rejects_non_dict() -> None:
    is_valid, error_code, reason = validate_flight_event_dict("not_a_dict")  # type: ignore[arg-type]
    assert is_valid is False
    assert error_code == "INVALID_TYPE"


def test_validate_flight_event_rejects_missing_envelope_fields() -> None:
    event = generate_synthetic_flight_event(seed=1).to_dict()
    del event["event_id"]
    is_valid, error_code, reason = validate_flight_event_dict(event)
    assert is_valid is False
    assert error_code == "MISSING_ENVELOPE"
    assert "event_id" in str(reason)


@pytest.mark.parametrize("empty_val", ["", "   ", "\t\n", None])
def test_validate_flight_event_rejects_empty_or_whitespace_event_id(empty_val: str | None) -> None:
    event = generate_synthetic_flight_event(seed=1).to_dict()
    event["event_id"] = empty_val
    is_valid, error_code, reason = validate_flight_event_dict(event)
    assert is_valid is False
    assert error_code in ("INVALID_EVENT_ID", "INVALID_TYPE")


def test_validate_flight_event_rejects_unsupported_version() -> None:
    event = generate_synthetic_flight_event(seed=1).to_dict()
    event["schema_version"] = "2.0.0"
    is_valid, error_code, reason = validate_flight_event_dict(event)
    assert is_valid is False
    assert error_code == "UNSUPPORTED_VERSION"


def test_validate_flight_event_rejects_invalid_timestamp() -> None:
    event = generate_synthetic_flight_event(seed=1).to_dict()
    event["event_timestamp"] = "not-a-timestamp"
    is_valid, error_code, reason = validate_flight_event_dict(event)
    assert is_valid is False
    assert error_code == "INVALID_TIMESTAMP"


def test_validate_flight_event_rejects_missing_flight_fields() -> None:
    event = generate_synthetic_flight_event(seed=1).to_dict()
    del event["carrier"]
    del event["distance"]
    is_valid, error_code, reason = validate_flight_event_dict(event)
    assert is_valid is False
    assert error_code == "MISSING_FIELDS"
    assert "carrier" in str(reason)
    assert "distance" in str(reason)


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_code"),
    [
        ("crs_dep_hour", 25, "INVALID_DEP_TIME"),
        ("crs_dep_minute", 60, "INVALID_DEP_TIME"),
        ("crs_arr_hour", -1, "INVALID_ARR_TIME"),
        ("month", 13, "INVALID_CALENDAR"),
        ("day_of_month", 32, "INVALID_CALENDAR"),
        ("day_of_week", 8, "INVALID_CALENDAR"),
        ("is_weekend", 2, "INVALID_WEEKEND"),
        ("crs_elapsed_time", 0.0, "INVALID_ELAPSED_TIME"),
        ("crs_elapsed_time", -10.0, "INVALID_ELAPSED_TIME"),
        ("distance", 0.0, "INVALID_DISTANCE"),
        ("distance", -200.0, "INVALID_DISTANCE"),
        ("carrier", "TOOLONG", "INVALID_CARRIER"),
        ("origin", "12", "INVALID_ORIGIN"),
        ("dest", "1234", "INVALID_DEST"),
    ],
)
def test_validate_flight_event_rejects_invalid_bounds(
    field: str, bad_value: int | float | str, expected_code: str
) -> None:
    event = generate_synthetic_flight_event(seed=1).to_dict()
    event[field] = bad_value
    is_valid, error_code, reason = validate_flight_event_dict(event)
    assert is_valid is False
    assert error_code == expected_code


def test_validate_flight_event_rejects_equal_origin_and_dest() -> None:
    event = generate_synthetic_flight_event(seed=1).to_dict()
    event["dest"] = event["origin"]
    is_valid, error_code, reason = validate_flight_event_dict(event)
    assert is_valid is False
    assert error_code == "EQUAL_ORIGIN_DEST"


@pytest.mark.parametrize("leakage_col", list(PROHIBITED_LEAKAGE_COLUMNS))
def test_validate_flight_event_rejects_all_leakage_columns(leakage_col: str) -> None:
    event = generate_synthetic_flight_event(seed=1).to_dict()
    event[leakage_col] = 15.0
    is_valid, error_code, reason = validate_flight_event_dict(event)
    assert is_valid is False
    assert error_code == "LEAKAGE_DETECTED"
    assert leakage_col.lower() in str(reason).lower()


def test_spark_schemas_are_valid_struct_types() -> None:
    event_schema = get_spark_flight_event_schema()
    dlq_schema = get_spark_dead_letter_schema()

    assert isinstance(event_schema, StructType)
    assert isinstance(dlq_schema, StructType)

    # Check key columns in event schema
    field_names = set(event_schema.fieldNames())
    assert "event_id" in field_names
    assert "schema_version" in field_names
    assert "carrier" in field_names
    assert "distance" in field_names
    assert "_corrupt_record" in field_names

    # Check key columns in DLQ schema
    dlq_names = set(dlq_schema.fieldNames())
    assert "event_id" in dlq_names
    assert "raw_payload" in dlq_names
    assert "error_code" in dlq_names
    assert "rejection_reason" in dlq_names
    assert "rejection_timestamp" in dlq_names


@pytest.mark.parametrize("leakage_col", list(PROHIBITED_LEAKAGE_COLUMNS))
def test_contains_prohibited_leakage_detects_each_column_and_nested(leakage_col: str) -> None:
    # 1. Top-level detection in dict
    assert contains_prohibited_leakage({leakage_col: 10.0}) is True
    # 2. Case-insensitivity
    assert contains_prohibited_leakage({leakage_col.upper(): 10.0}) is True
    # 3. Nested inside another dictionary
    assert contains_prohibited_leakage({"source": {leakage_col: "val"}}) is True
    # 4. JSON string format
    assert contains_prohibited_leakage(f'{{"{leakage_col}": 100}}') is True


def test_contains_prohibited_leakage_ignores_values_and_clean_payloads() -> None:
    # Clean payload
    assert contains_prohibited_leakage({"carrier": "AA", "origin": "DFW"}) is False
    # Value contains leakage substring but key is clean
    assert contains_prohibited_leakage({"notes": "Flight dep_delay was averted"}) is False
    # Malformed JSON or None
    assert contains_prohibited_leakage(None) is False
    assert contains_prohibited_leakage("{unparseable_json") is False
