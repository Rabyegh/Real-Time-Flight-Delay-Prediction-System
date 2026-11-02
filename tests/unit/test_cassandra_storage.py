"""Unit tests for the Cassandra persistence layer (CassandraStorage).

Validates schema constraints, connection lifecycle, prepared statements,
idempotent write semantics, bounds enforcement, and driver error propagation
using deterministic mocks without requiring Docker or a live Cassandra instance.
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from cassandra import ConsistencyLevel, DriverException
from cassandra.query import PreparedStatement
from cassandra.util import Date

from flight_delay.storage.cassandra import (
    MAX_QUERY_LIMIT,
    CassandraConfig,
    CassandraStorage,
    PredictionRecord,
    StorageConnectionError,
    StorageQueryError,
    StorageValidationError,
    StorageWriteError,
)


@pytest.fixture
def valid_record() -> PredictionRecord:
    """Return a representative PredictionRecord."""
    return PredictionRecord(
        event_id="test-event-uuid-12345",
        carrier="AA",
        flight_number="100",
        origin="JFK",
        dest="LAX",
        event_time=datetime.datetime(2026, 10, 23, 10, 30, 0, tzinfo=datetime.UTC),
        prediction_time=datetime.datetime(2026, 10, 23, 10, 35, 0, tzinfo=datetime.UTC),
        delay_probability=0.256006,
        predicted_delayed=False,
        model_version="baseline-v1",
    )


@pytest.fixture
def mock_session() -> MagicMock:
    """Create a mock Cassandra Session that tracks prepared statements and queries."""
    session = MagicMock()
    session.is_shutdown = False

    # Simulate session.prepare returning PreparedStatement-like objects
    def fake_prepare(cql: str) -> MagicMock:
        stmt = MagicMock(spec=PreparedStatement)
        stmt.query_string = cql.strip()
        return stmt

    session.prepare.side_effect = fake_prepare
    return session


@pytest.fixture
def mock_cluster(mock_session: MagicMock) -> MagicMock:
    """Create a mock Cassandra Cluster."""
    cluster = MagicMock()
    cluster.is_shutdown = False
    cluster.connect.return_value = mock_session
    return cluster


# -----------------------------------------------------------------------------
# 1. Configuration & PredictionRecord Tests
# -----------------------------------------------------------------------------


def test_cassandra_config_defaults() -> None:
    """Verify CassandraConfig default settings."""
    cfg = CassandraConfig()
    assert cfg.contact_points == ("localhost",)
    assert cfg.port == 9042
    assert cfg.keyspace == "flight_delay"
    assert cfg.connect_timeout == 10.0
    assert cfg.read_timeout == 10.0
    assert cfg.max_retries == 3
    assert cfg.consistency_level == ConsistencyLevel.LOCAL_ONE


def test_prediction_record_normalization() -> None:
    """Verify PredictionRecord normalizes string timestamps and derives prediction_date."""
    rec = PredictionRecord(
        event_id="evt-001",
        carrier="DL",
        flight_number="200",
        origin="ORD",
        dest="ATL",
        event_time="2026-10-23T14:00:00Z",  # type: ignore[arg-type]
        prediction_time="2026-10-23T14:05:00Z",  # type: ignore[arg-type]
        delay_probability=0.65,
        predicted_delayed=1,  # type: ignore[arg-type]
        model_version="baseline-v1",
    )
    assert isinstance(rec.event_time, datetime.datetime)
    assert isinstance(rec.prediction_time, datetime.datetime)
    assert rec.prediction_date == datetime.date(2026, 10, 23)
    assert rec.delay_probability == 0.65
    assert rec.predicted_delayed is True

    # Dictionary conversion and reconstruction
    d = rec.to_dict()
    assert d["event_id"] == "evt-001"
    assert d["prediction_date"] == "2026-10-23"

    reconstructed = PredictionRecord.from_dict(d)
    assert reconstructed.event_id == rec.event_id
    assert reconstructed.delay_probability == rec.delay_probability
    assert reconstructed.prediction_date == rec.prediction_date


@pytest.mark.parametrize(
    ("bad_field", "bad_val", "expected_err"),
    [
        ("event_id", "", "event_id must be a non-empty string"),
        ("event_id", "   ", "event_id must be a non-empty string"),
        ("carrier", "TOOLONG", "Invalid carrier code"),
        ("carrier", "1", "Invalid carrier code"),
        ("origin", "JF", "Invalid origin airport IATA code"),
        ("dest", "LAXX", "Invalid destination airport IATA code"),
        ("delay_probability", -0.01, "delay_probability must be in range"),
        ("delay_probability", 1.01, "delay_probability must be in range"),
        ("delay_probability", float("nan"), "must be a finite number"),
        ("delay_probability", float("inf"), "must be a finite number"),
        ("model_version", "", "model_version must be a non-empty string"),
    ],
)
def test_prediction_record_validation_failures(
    valid_record: PredictionRecord,
    bad_field: str,
    bad_val: Any,
    expected_err: str,
) -> None:
    """Ensure invalid record fields raise StorageValidationError."""
    d = valid_record.to_dict()
    d[bad_field] = bad_val

    if bad_field in ("delay_probability", "event_id", "model_version"):
        # __post_init__ or validate should fail
        with pytest.raises(StorageValidationError, match=expected_err):
            rec = PredictionRecord.from_dict(d)
            rec.validate()
    else:
        rec = PredictionRecord.from_dict(d)
        with pytest.raises(StorageValidationError, match=expected_err):
            rec.validate()


# -----------------------------------------------------------------------------
# 2. Connection Lifecycle & Prepared Statements
# -----------------------------------------------------------------------------


def test_connect_prepares_all_statements(mock_cluster: MagicMock, mock_session: MagicMock) -> None:
    """Verify connect() initializes Cluster, connects to keyspace, and prepares CQL statements."""
    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage(contact_points=["127.0.0.1"], port=9042, keyspace="flight_delay")
        assert not storage.is_connected

        storage.connect()
        assert storage.is_connected
        mock_cluster.connect.assert_called_once_with("flight_delay")

        # 4 prepared statements: 2 inserts + 2 selects
        assert mock_session.prepare.call_count == 4


def test_connect_failure_cleans_up_safely(mock_cluster: MagicMock) -> None:
    """Verify connection errors cleanly shutdown cluster and raise StorageConnectionError."""
    mock_cluster.connect.side_effect = DriverException("Cluster unreachable")

    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        with pytest.raises(StorageConnectionError, match="Failed to connect to Cassandra cluster"):
            storage.connect()

        assert not storage.is_connected
        mock_cluster.shutdown.assert_called_once()


def test_close_lifecycle_and_context_manager(
    mock_cluster: MagicMock, mock_session: MagicMock
) -> None:
    """Verify close() and context manager cleanly release resources."""
    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        with CassandraStorage() as storage:
            assert storage.is_connected

        assert not storage.is_connected
        mock_session.shutdown.assert_called_once()
        mock_cluster.shutdown.assert_called_once()

        # Idempotent second close
        storage.close()
        assert not storage.is_connected


# -----------------------------------------------------------------------------
# 3. Write Operations & Parameterized Execution
# -----------------------------------------------------------------------------


def test_upsert_prediction_by_event_uses_parameters(
    mock_cluster: MagicMock, mock_session: MagicMock, valid_record: PredictionRecord
) -> None:
    """Verify upsert_prediction_by_event executes prepared statement with exact tuple parameters."""
    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        storage.connect()

        storage.upsert_prediction_by_event(valid_record)

        mock_session.execute.assert_called_once()
        stmt, params = mock_session.execute.call_args[0]

        # Verify prepared statement used (not raw string)
        assert isinstance(stmt, PreparedStatement) or hasattr(stmt, "query_string")
        assert "predictions_by_event" in stmt.query_string

        # Verify parameter values and order
        assert params[0] == valid_record.event_id
        assert params[1] == valid_record.carrier
        assert params[2] == valid_record.flight_number
        assert params[3] == valid_record.origin
        assert params[4] == valid_record.dest
        assert params[5] == valid_record.event_time
        assert params[6] == valid_record.prediction_time
        assert isinstance(params[7], Date)
        assert params[8] == valid_record.delay_probability
        assert params[9] == valid_record.predicted_delayed
        assert params[10] == valid_record.model_version


def test_upsert_prediction_by_airport_uses_parameters(
    mock_cluster: MagicMock, mock_session: MagicMock, valid_record: PredictionRecord
) -> None:
    """Verify upsert_prediction_by_airport executes with origin and date-bucket partition key."""
    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        storage.connect()

        storage.upsert_prediction_by_airport(valid_record)

        mock_session.execute.assert_called_once()
        stmt, params = mock_session.execute.call_args[0]

        assert "predictions_by_airport" in stmt.query_string
        # Partition key components: (airport, prediction_date)
        assert params[0] == valid_record.origin
        assert isinstance(params[1], Date)
        assert params[2] == valid_record.prediction_time
        assert params[3] == valid_record.event_id


def test_repeated_writes_use_identical_primary_keys(
    mock_cluster: MagicMock, mock_session: MagicMock, valid_record: PredictionRecord
) -> None:
    """Ensure replaying the same record executes writes with identical primary keys."""
    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        storage.connect()

        # Write once
        storage.upsert_prediction_by_event(valid_record)
        first_call_params = mock_session.execute.call_args_list[0][0][1]

        # Write twice (replayed event)
        storage.upsert_prediction_by_event(valid_record)
        second_call_params = mock_session.execute.call_args_list[1][0][1]

        # Primary key (event_id) and payload must be identical
        assert first_call_params[0] == second_call_params[0] == valid_record.event_id
        assert first_call_params == second_call_params


def test_upsert_prediction_dual_write_and_batch(
    mock_cluster: MagicMock, mock_session: MagicMock, valid_record: PredictionRecord
) -> None:
    """Verify upsert_prediction and upsert_predictions_batch persist to both tables."""
    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        storage.connect()

        # Single dual write
        storage.upsert_prediction(valid_record)
        assert mock_session.execute.call_count == 2

        mock_session.reset_mock()

        # Batch write of 3 records -> 6 executions
        records = [valid_record, valid_record, valid_record]
        storage.upsert_predictions_batch(records)
        assert mock_session.execute.call_count == 6


# -----------------------------------------------------------------------------
# 4. Read Operations & Bounds Enforcement
# -----------------------------------------------------------------------------


def test_get_prediction_by_event_success(
    mock_cluster: MagicMock, mock_session: MagicMock, valid_record: PredictionRecord
) -> None:
    """Verify get_prediction_by_event successfully maps row to PredictionRecord."""
    mock_row = MagicMock()
    mock_row.event_id = valid_record.event_id
    mock_row.carrier = valid_record.carrier
    mock_row.flight_number = valid_record.flight_number
    mock_row.origin = valid_record.origin
    mock_row.dest = valid_record.dest
    mock_row.event_time = valid_record.event_time
    mock_row.prediction_time = valid_record.prediction_time
    mock_row.prediction_date = Date(valid_record.prediction_date)
    mock_row.delay_probability = valid_record.delay_probability
    mock_row.predicted_delayed = valid_record.predicted_delayed
    mock_row.model_version = valid_record.model_version

    mock_result = MagicMock()
    mock_result.one.return_value = mock_row
    mock_session.execute.return_value = mock_result

    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        res = storage.get_prediction_by_event(valid_record.event_id)

        assert res is not None
        assert res.event_id == valid_record.event_id
        assert res.delay_probability == valid_record.delay_probability
        assert res.origin == "JFK"


def test_get_prediction_by_event_not_found(
    mock_cluster: MagicMock, mock_session: MagicMock
) -> None:
    """Verify get_prediction_by_event returns None when record does not exist."""
    mock_result = MagicMock()
    mock_result.one.return_value = None
    mock_session.execute.return_value = mock_result

    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        assert storage.get_prediction_by_event("missing-id") is None


def test_get_prediction_by_event_empty_id_raises_error(mock_cluster: MagicMock) -> None:
    """Ensure querying with empty event ID raises StorageValidationError."""
    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        with pytest.raises(StorageValidationError, match="event_id must be a non-empty string"):
            storage.get_prediction_by_event("   ")


def test_get_predictions_by_airport_success(
    mock_cluster: MagicMock, mock_session: MagicMock, valid_record: PredictionRecord
) -> None:
    """Verify get_predictions_by_airport retrieves records within date bucket."""
    mock_row = MagicMock()
    mock_row.event_id = valid_record.event_id
    mock_row.carrier = valid_record.carrier
    mock_row.flight_number = valid_record.flight_number
    mock_row.origin = "JFK"
    mock_row.dest = valid_record.dest
    mock_row.event_time = valid_record.event_time
    mock_row.prediction_time = valid_record.prediction_time
    mock_row.prediction_date = Date(valid_record.prediction_date)
    mock_row.delay_probability = valid_record.delay_probability
    mock_row.predicted_delayed = valid_record.predicted_delayed
    mock_row.model_version = valid_record.model_version

    mock_session.execute.return_value = [mock_row]

    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        results = storage.get_predictions_by_airport(
            airport="JFK",
            date="2026-10-23",
            limit=25,
        )

        assert len(results) == 1
        assert results[0].origin == "JFK"
        assert results[0].prediction_date == datetime.date(2026, 10, 23)

        # Check call arguments
        _, params = mock_session.execute.call_args[0]
        assert params[0] == "JFK"
        assert isinstance(params[1], Date)
        assert params[2] == 25


@pytest.mark.parametrize(
    ("bad_airport", "bad_date", "bad_limit", "expected_err"),
    [
        ("", "2026-10-23", 50, "airport code must be a non-empty string"),
        ("TOOLONG", "2026-10-23", 50, "Invalid airport IATA code"),
        ("12", "2026-10-23", 50, "Invalid airport IATA code"),
        ("JFK", "2026-13-45", 50, "Invalid date format"),
        ("JFK", 12345, 50, "Unsupported date type"),
        ("JFK", "2026-10-23", 0, f"Query limit must be between 1 and {MAX_QUERY_LIMIT}"),
        ("JFK", "2026-10-23", -10, f"Query limit must be between 1 and {MAX_QUERY_LIMIT}"),
        ("JFK", "2026-10-23", 501, f"Query limit must be between 1 and {MAX_QUERY_LIMIT}"),
    ],
)
def test_get_predictions_by_airport_validation_errors(
    mock_cluster: MagicMock,
    bad_airport: str,
    bad_date: Any,
    bad_limit: int,
    expected_err: str,
) -> None:
    """Ensure invalid airport, date, or query limit parameters raise StorageValidationError."""
    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        with pytest.raises(StorageValidationError, match=expected_err):
            storage.get_predictions_by_airport(
                airport=bad_airport,
                date=bad_date,
                limit=bad_limit,
            )


# -----------------------------------------------------------------------------
# 5. Driver Error Handling & Propagation
# -----------------------------------------------------------------------------


def test_write_error_propagated_with_clear_exception(
    mock_cluster: MagicMock, mock_session: MagicMock, valid_record: PredictionRecord
) -> None:
    """Verify Cassandra write failures raise StorageWriteError with context."""
    mock_session.execute.side_effect = DriverException("Write timed out on node 127.0.0.1")

    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        with pytest.raises(StorageWriteError, match="Failed to upsert prediction_by_event"):
            storage.upsert_prediction_by_event(valid_record)


def test_query_error_propagated_with_clear_exception(
    mock_cluster: MagicMock, mock_session: MagicMock
) -> None:
    """Verify Cassandra query failures raise StorageQueryError with context."""
    mock_session.execute.side_effect = DriverException("Coordinator unavailable")

    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        with pytest.raises(StorageQueryError, match="Failed to query prediction_by_event"):
            storage.get_prediction_by_event("any-event-id")


def test_upsert_predictions_batch_partial_failure(
    mock_cluster: MagicMock, mock_session: MagicMock, valid_record: PredictionRecord
) -> None:
    """Verify that a failure on record i in a batch raises StorageWriteError with index."""
    # First record writes to event & airport (2 calls); second record fails on event write
    mock_session.execute.side_effect = [
        None,  # rec 0: event write succeeds
        None,  # rec 0: airport write succeeds
        DriverException("Cassandra node timeout on event write"),  # rec 1: fails
    ]

    rec0 = valid_record
    rec1 = PredictionRecord(
        event_id="test-event-uuid-67890",
        carrier="DL",
        flight_number="200",
        origin="ORD",
        dest="DFW",
        event_time=datetime.datetime(2026, 10, 23, 11, 0, 0, tzinfo=datetime.UTC),
        prediction_time=datetime.datetime(2026, 10, 23, 11, 5, 0, tzinfo=datetime.UTC),
        delay_probability=0.75,
        predicted_delayed=True,
        model_version="baseline-v1",
    )

    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        with pytest.raises(StorageWriteError) as exc_info:
            storage.upsert_predictions_batch([rec0, rec1])

        err_msg = str(exc_info.value)
        assert "batch index 1" in err_msg
        assert "test-event-uuid-67890" in err_msg


def test_retry_after_partial_failure_is_idempotent(
    mock_cluster: MagicMock, mock_session: MagicMock, valid_record: PredictionRecord
) -> None:
    """Verify that re-executing a failed batch safely upserts all records idempotently."""
    mock_session.execute.side_effect = None

    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        # Simulate retry of valid_record batch
        storage.upsert_predictions_batch([valid_record])
        storage.upsert_predictions_batch([valid_record])

        assert mock_session.execute.call_count == 4
        # First attempt and second attempt target identical primary keys
        call1_event = mock_session.execute.call_args_list[0][0][1]
        call2_event = mock_session.execute.call_args_list[2][0][1]
        assert call1_event[0] == call2_event[0] == valid_record.event_id

        call1_airport = mock_session.execute.call_args_list[1][0][1]
        call2_airport = mock_session.execute.call_args_list[3][0][1]
        # (airport, prediction_date, prediction_time, event_id)
        assert call1_airport[0:4] == call2_airport[0:4]


def test_timezone_naive_and_aware_timestamp_normalization() -> None:
    """Verify that naive datetimes and non-UTC datetimes are normalized to UTC."""
    # 1. Naive datetime
    naive_event = datetime.datetime(2026, 6, 15, 10, 30, 0)
    naive_pred = datetime.datetime(2026, 6, 15, 10, 35, 0)
    rec_naive = PredictionRecord(
        event_id="test-naive",
        carrier="AA",
        flight_number="1",
        origin="JFK",
        dest="LAX",
        event_time=naive_event,
        prediction_time=naive_pred,
        delay_probability=0.1,
        predicted_delayed=False,
        model_version="v1",
    )
    assert rec_naive.event_time.tzinfo == datetime.UTC
    assert rec_naive.prediction_time.tzinfo == datetime.UTC
    assert rec_naive.prediction_date == datetime.date(2026, 6, 15)

    # 2. Non-UTC timezone (e.g. UTC+5)
    tz_plus_5 = datetime.timezone(datetime.timedelta(hours=5))
    dt_plus_5 = datetime.datetime(2026, 6, 15, 3, 0, 0, tzinfo=tz_plus_5)  # 2026-06-14 22:00 UTC
    rec_tz = PredictionRecord(
        event_id="test-tz",
        carrier="AA",
        flight_number="1",
        origin="JFK",
        dest="LAX",
        event_time=dt_plus_5,
        prediction_time=dt_plus_5,
        delay_probability=0.1,
        predicted_delayed=False,
        model_version="v1",
    )
    assert rec_tz.prediction_time.tzinfo == datetime.UTC
    # 22:00 on June 14 UTC -> date bucket must be June 14
    assert rec_tz.prediction_date == datetime.date(2026, 6, 14)


def test_prediction_record_mismatched_prediction_date_raises_error() -> None:
    """Ensure an explicit prediction_date differing from prediction_time.date() raises error."""
    rec = PredictionRecord(
        event_id="test-date-mismatch",
        carrier="AA",
        flight_number="100",
        origin="JFK",
        dest="LAX",
        event_time=datetime.datetime(2026, 6, 15, 10, 0, 0, tzinfo=datetime.UTC),
        prediction_time=datetime.datetime(2026, 6, 15, 10, 5, 0, tzinfo=datetime.UTC),
        prediction_date=datetime.date(2026, 1, 1),  # Mismatched date
        delay_probability=0.2,
        predicted_delayed=False,
        model_version="baseline-v1",
    )
    with pytest.raises(StorageValidationError, match="must match prediction_time UTC date"):
        rec.validate()


def test_upsert_prediction_partial_failure_and_replay_behavior(
    mock_cluster: MagicMock, mock_session: MagicMock, valid_record: PredictionRecord
) -> None:
    """Verify that failure in airport write after event write raises error and retry succeeds."""
    # Attempt 1: event write succeeds, airport write fails
    mock_session.execute.side_effect = [
        None,  # event write succeeds
        DriverException("Airport table write timed out"),  # airport write fails
    ]

    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        with pytest.raises(StorageWriteError, match="Failed to upsert prediction_by_airport"):
            storage.upsert_prediction(valid_record)

        # Attempt 2: retry of the exact same record
        mock_session.execute.side_effect = [
            None,  # event write idempotent overwrite succeeds
            None,  # airport write succeeds
        ]
        storage.upsert_prediction(valid_record)

        # Total 4 execute calls: 2 from first attempt, 2 from second attempt
        assert mock_session.execute.call_count == 4
        # Verify primary keys match across both attempts
        event_call_1 = mock_session.execute.call_args_list[0][0][1]
        event_call_2 = mock_session.execute.call_args_list[2][0][1]
        assert event_call_1[0] == event_call_2[0] == valid_record.event_id


def test_event_id_sha256_format_contract_preserved(
    mock_cluster: MagicMock, mock_session: MagicMock
) -> None:
    """Verify 32-character SHA-256 event ID from contract is accepted and preserved exactly."""
    from flight_delay.ingestion.contract import generate_event_id

    event_id_32 = generate_event_id("AA", "100", "JFK", "LAX", "2026-06-15", "1030")
    assert len(event_id_32) == 32
    assert all(c in "0123456789abcdef" for c in event_id_32)

    rec = PredictionRecord(
        event_id=event_id_32,
        carrier="AA",
        flight_number="100",
        origin="JFK",
        dest="LAX",
        event_time=datetime.datetime(2026, 6, 15, 10, 30, 0, tzinfo=datetime.UTC),
        prediction_time=datetime.datetime(2026, 6, 15, 10, 30, 0, tzinfo=datetime.UTC),
        delay_probability=0.15,
        predicted_delayed=False,
        model_version="baseline-v1",
    )
    rec.validate()
    assert rec.event_id == event_id_32

    with patch("flight_delay.storage.cassandra.Cluster", return_value=mock_cluster):
        storage = CassandraStorage()
        storage.upsert_prediction(rec)
        first_call = mock_session.execute.call_args_list[0][0][1]
        assert first_call[0] == event_id_32
