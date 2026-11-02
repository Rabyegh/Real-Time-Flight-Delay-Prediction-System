"""Cassandra persistence layer for real-time flight delay predictions.

Delivery Guarantees & Duplicate Semantics:
- Deterministic primary keys guarantee idempotent upserts for both point lookups
  (predictions_by_event) and airport timelines (predictions_by_airport).
- Replays of the same event with identical payload result in identical logical predictions.
- End-to-end processing is at-least-once, NOT end-to-end exactly-once.
- Time-bucketing by calendar date in predictions_by_airport prevents unbounded partition
  growth while enabling pre-sorted, bounded range queries without ALLOW FILTERING.
"""

from __future__ import annotations

import datetime
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from cassandra import ConsistencyLevel, DriverException
from cassandra.cluster import Cluster, Session
from cassandra.query import PreparedStatement
from cassandra.util import Date

LOGGER = logging.getLogger(__name__)

IATA_CODE_REGEX = re.compile(r"^[A-Z0-9]{3}$")
CARRIER_CODE_REGEX = re.compile(r"^[A-Z0-9]{2,3}$")
MAX_QUERY_LIMIT = 500
DEFAULT_QUERY_LIMIT = 50


class CassandraStorageError(RuntimeError):
    """Base exception for Cassandra storage operations."""


class StorageConnectionError(CassandraStorageError):
    """Raised when connecting to the Cassandra cluster or keyspace fails."""


class StorageWriteError(CassandraStorageError):
    """Raised when persisting a prediction record fails."""


class StorageQueryError(CassandraStorageError):
    """Raised when querying prediction records fails."""


class StorageValidationError(CassandraStorageError, ValueError):
    """Raised when a prediction record or query parameter fails validation."""


@dataclass(frozen=True)
class CassandraConfig:
    """Configuration options for Cassandra connection and execution."""

    contact_points: tuple[str, ...] = ("localhost",)
    port: int = 9042
    keyspace: str = "flight_delay"
    connect_timeout: float = 10.0
    read_timeout: float = 10.0
    max_retries: int = 3
    consistency_level: ConsistencyLevel = ConsistencyLevel.LOCAL_ONE
    protocol_version: int | None = None


@dataclass(frozen=True)
class PredictionRecord:
    """Canonical flight delay prediction record for database storage and retrieval."""

    event_id: str
    carrier: str
    flight_number: str | None
    origin: str
    dest: str
    event_time: datetime.datetime
    prediction_time: datetime.datetime
    delay_probability: float
    predicted_delayed: bool
    model_version: str
    prediction_date: datetime.date = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Normalize types, timestamps, and date buckets to UTC."""
        # Normalize event_time to timezone-aware UTC datetime
        if isinstance(self.event_time, str):
            dt_event = datetime.datetime.fromisoformat(self.event_time.replace("Z", "+00:00"))
        elif isinstance(self.event_time, datetime.datetime):
            dt_event = self.event_time
        else:
            dt_event = self.event_time

        if isinstance(dt_event, datetime.datetime):
            if dt_event.tzinfo is None:
                dt_event = dt_event.replace(tzinfo=datetime.UTC)
            else:
                dt_event = dt_event.astimezone(datetime.UTC)
            object.__setattr__(self, "event_time", dt_event)

        # Normalize prediction_time to timezone-aware UTC datetime
        if isinstance(self.prediction_time, str):
            dt_pred = datetime.datetime.fromisoformat(self.prediction_time.replace("Z", "+00:00"))
        elif isinstance(self.prediction_time, datetime.datetime):
            dt_pred = self.prediction_time
        else:
            dt_pred = self.prediction_time

        if isinstance(dt_pred, datetime.datetime):
            if dt_pred.tzinfo is None:
                dt_pred = dt_pred.replace(tzinfo=datetime.UTC)
            else:
                dt_pred = dt_pred.astimezone(datetime.UTC)
            object.__setattr__(self, "prediction_time", dt_pred)

        # Derive prediction_date in UTC if missing
        if self.prediction_date is None:
            if isinstance(self.prediction_time, datetime.datetime):
                derived_date = self.prediction_time.date()
            else:
                derived_date = datetime.date.fromisoformat(str(self.prediction_time)[:10])
            object.__setattr__(self, "prediction_date", derived_date)
        elif isinstance(self.prediction_date, str):
            object.__setattr__(
                self, "prediction_date", datetime.date.fromisoformat(self.prediction_date)
            )

        # Ensure delay_probability is float
        object.__setattr__(self, "delay_probability", float(self.delay_probability))

        # Ensure predicted_delayed is boolean
        object.__setattr__(self, "predicted_delayed", bool(self.predicted_delayed))

    def validate(self) -> None:
        """Validate all required fields, ranges, and patterns."""
        if not self.event_id or not str(self.event_id).strip():
            raise StorageValidationError("event_id must be a non-empty string.")

        carrier_upper = str(self.carrier).strip().upper()
        if not CARRIER_CODE_REGEX.match(carrier_upper):
            raise StorageValidationError(f"Invalid carrier code: '{self.carrier}'")

        origin_upper = str(self.origin).strip().upper()
        dest_upper = str(self.dest).strip().upper()
        if not IATA_CODE_REGEX.match(origin_upper):
            raise StorageValidationError(f"Invalid origin airport IATA code: '{self.origin}'")
        if not IATA_CODE_REGEX.match(dest_upper):
            raise StorageValidationError(f"Invalid destination airport IATA code: '{self.dest}'")

        if not math.isfinite(self.delay_probability):
            raise StorageValidationError("delay_probability must be a finite number.")
        if not (0.0 <= self.delay_probability <= 1.0):
            raise StorageValidationError(
                f"delay_probability must be in range [0.0, 1.0], got {self.delay_probability}"
            )

        if not isinstance(self.event_time, datetime.datetime):
            raise StorageValidationError("event_time must be a valid datetime.")
        if not isinstance(self.prediction_time, datetime.datetime):
            raise StorageValidationError("prediction_time must be a valid datetime.")
        if not isinstance(self.prediction_date, datetime.date):
            raise StorageValidationError("prediction_date must be a valid date.")
        if self.prediction_date != self.prediction_time.date():
            raise StorageValidationError(
                f"prediction_date ({self.prediction_date}) must match "
                f"prediction_time UTC date ({self.prediction_time.date()})."
            )

        if not self.model_version or not str(self.model_version).strip():
            raise StorageValidationError("model_version must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction record to dictionary."""
        d = asdict(self)
        d["event_time"] = self.event_time.isoformat()
        d["prediction_time"] = self.prediction_time.isoformat()
        d["prediction_date"] = self.prediction_date.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionRecord:
        """Create a PredictionRecord from dictionary."""
        return cls(
            event_id=data["event_id"],
            carrier=data["carrier"],
            flight_number=data.get("flight_number"),
            origin=data["origin"],
            dest=data["dest"],
            event_time=data["event_time"],
            prediction_time=data["prediction_time"],
            delay_probability=data["delay_probability"],
            predicted_delayed=data["predicted_delayed"],
            model_version=data.get("model_version", "baseline-v1"),
            prediction_date=data.get("prediction_date"),
        )


class CassandraStorage:
    """Thread-safe Cassandra storage manager for flight delay predictions."""

    CQL_INSERT_EVENT = """
        INSERT INTO predictions_by_event (
            event_id, carrier, flight_number, origin, dest,
            event_time, prediction_time, prediction_date,
            delay_probability, predicted_delayed, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    CQL_INSERT_AIRPORT = """
        INSERT INTO predictions_by_airport (
            airport, prediction_date, prediction_time, event_id,
            carrier, flight_number, dest, event_time,
            delay_probability, predicted_delayed, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    CQL_SELECT_EVENT = """
        SELECT event_id, carrier, flight_number, origin, dest,
               event_time, prediction_time, prediction_date,
               delay_probability, predicted_delayed, model_version
        FROM predictions_by_event
        WHERE event_id = ?;
    """

    CQL_SELECT_AIRPORT = """
        SELECT airport, prediction_date, prediction_time, event_id,
               carrier, flight_number, dest, event_time,
               delay_probability, predicted_delayed, model_version
        FROM predictions_by_airport
        WHERE airport = ? AND prediction_date = ?
        LIMIT ?;
    """

    def __init__(
        self,
        config: CassandraConfig | None = None,
        contact_points: tuple[str, ...] | list[str] | None = None,
        port: int | None = None,
        keyspace: str | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
    ) -> None:
        """Initialize CassandraStorage with configuration.

        Does not connect automatically; call connect() or use as context manager.
        """
        cfg = config or CassandraConfig()
        points = tuple(contact_points) if contact_points is not None else cfg.contact_points
        resolved_port = port if port is not None else cfg.port
        resolved_keyspace = keyspace if keyspace is not None else cfg.keyspace
        c_timeout = connect_timeout if connect_timeout is not None else cfg.connect_timeout
        r_timeout = read_timeout if read_timeout is not None else cfg.read_timeout

        self.config = CassandraConfig(
            contact_points=points,
            port=resolved_port,
            keyspace=resolved_keyspace,
            connect_timeout=c_timeout,
            read_timeout=r_timeout,
            max_retries=cfg.max_retries,
            consistency_level=cfg.consistency_level,
            protocol_version=cfg.protocol_version,
        )

        self._cluster: Cluster | None = None
        self._session: Session | None = None
        self._stmt_insert_event: PreparedStatement | None = None
        self._stmt_insert_airport: PreparedStatement | None = None
        self._stmt_select_event: PreparedStatement | None = None
        self._stmt_select_airport: PreparedStatement | None = None

    @property
    def is_connected(self) -> bool:
        """Return True if session is active and not closed."""
        return self._session is not None and not self._session.is_shutdown

    def connect(self) -> None:
        """Establish cluster connection, open session, and prepare CQL statements."""
        if self.is_connected:
            return

        LOGGER.info(
            "Connecting to Cassandra cluster at %s:%d (keyspace: %s)",
            self.config.contact_points,
            self.config.port,
            self.config.keyspace,
        )

        try:
            self._cluster = Cluster(
                contact_points=list(self.config.contact_points),
                port=self.config.port,
                connect_timeout=self.config.connect_timeout,
                control_connection_timeout=self.config.connect_timeout,
                protocol_version=self.config.protocol_version,
            )
            self._session = self._cluster.connect(self.config.keyspace)
            self._session.default_timeout = self.config.read_timeout

            # Prepare statements
            self._stmt_insert_event = self._session.prepare(self.CQL_INSERT_EVENT)
            self._stmt_insert_airport = self._session.prepare(self.CQL_INSERT_AIRPORT)
            self._stmt_select_event = self._session.prepare(self.CQL_SELECT_EVENT)
            self._stmt_select_airport = self._session.prepare(self.CQL_SELECT_AIRPORT)

            LOGGER.info("Successfully connected to Cassandra keyspace '%s'", self.config.keyspace)

        except Exception as exc:
            self.close()
            raise StorageConnectionError(
                f"Failed to connect to Cassandra cluster at {self.config.contact_points}: {exc}"
            ) from exc

    def close(self) -> None:
        """Cleanly close session and cluster connections."""
        if self._session is not None and not self._session.is_shutdown:
            try:
                self._session.shutdown()
            except Exception as exc:
                LOGGER.warning("Error shutting down Cassandra session: %s", exc)
            finally:
                self._session = None

        if self._cluster is not None and not self._cluster.is_shutdown:
            try:
                self._cluster.shutdown()
            except Exception as exc:
                LOGGER.warning("Error shutting down Cassandra cluster: %s", exc)
            finally:
                self._cluster = None

        self._stmt_insert_event = None
        self._stmt_insert_airport = None
        self._stmt_select_event = None
        self._stmt_select_airport = None

    def __enter__(self) -> CassandraStorage:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _ensure_connected(self) -> Session:
        """Ensure connection is open or attempt connection."""
        if not self.is_connected:
            self.connect()
        assert self._session is not None
        return self._session

    def upsert_prediction_by_event(self, record: PredictionRecord | dict[str, Any]) -> None:
        """Upsert a prediction into predictions_by_event table.

        Idempotent write keyed by event_id.
        """
        rec = record if isinstance(record, PredictionRecord) else PredictionRecord.from_dict(record)
        rec.validate()

        session = self._ensure_connected()
        assert self._stmt_insert_event is not None

        params = (
            rec.event_id,
            rec.carrier,
            rec.flight_number,
            rec.origin,
            rec.dest,
            rec.event_time,
            rec.prediction_time,
            Date(rec.prediction_date),
            rec.delay_probability,
            rec.predicted_delayed,
            rec.model_version,
        )

        try:
            session.execute(self._stmt_insert_event, params)
        except DriverException as exc:
            raise StorageWriteError(
                f"Failed to upsert prediction_by_event for event_id '{rec.event_id}': {exc}"
            ) from exc

    def upsert_prediction_by_airport(self, record: PredictionRecord | dict[str, Any]) -> None:
        """Upsert a prediction into predictions_by_airport table.

        Idempotent write keyed by ((origin, prediction_date), prediction_time, event_id).
        """
        rec = record if isinstance(record, PredictionRecord) else PredictionRecord.from_dict(record)
        rec.validate()

        session = self._ensure_connected()
        assert self._stmt_insert_airport is not None

        params = (
            rec.origin,
            Date(rec.prediction_date),
            rec.prediction_time,
            rec.event_id,
            rec.carrier,
            rec.flight_number,
            rec.dest,
            rec.event_time,
            rec.delay_probability,
            rec.predicted_delayed,
            rec.model_version,
        )

        try:
            session.execute(self._stmt_insert_airport, params)
        except DriverException as exc:
            raise StorageWriteError(
                f"Failed to upsert prediction_by_airport for airport '{rec.origin}': {exc}"
            ) from exc

    def upsert_prediction(self, record: PredictionRecord | dict[str, Any]) -> None:
        """Upsert a prediction into both predictions_by_event and predictions_by_airport."""
        rec = record if isinstance(record, PredictionRecord) else PredictionRecord.from_dict(record)
        rec.validate()
        self.upsert_prediction_by_event(rec)
        self.upsert_prediction_by_airport(rec)

    def upsert_predictions_batch(self, records: list[PredictionRecord | dict[str, Any]]) -> None:
        """Upsert a collection of predictions into both query tables.

        Writes across predictions_by_event and predictions_by_airport are executed as
        independent parameterized CQL statements, NOT logged or unlogged cross-partition
        batches, avoiding anti-pattern cross-partition coordination overhead.
        Failures raise StorageWriteError detailing the failed record and table.
        Retries are safe and idempotent because writes on matching primary keys overwrite
        existing rows in place.
        """
        for idx, rec in enumerate(records):
            record_obj = (
                rec if isinstance(rec, PredictionRecord) else PredictionRecord.from_dict(rec)
            )
            try:
                self.upsert_prediction(record_obj)
            except Exception as exc:
                raise StorageWriteError(
                    f"Failed upserting prediction for event '{record_obj.event_id}' "
                    f"at batch index {idx}: {exc}"
                ) from exc

    def get_prediction_by_event(self, event_id: str) -> PredictionRecord | None:
        """Retrieve a prediction by its deterministic event ID.

        Returns None if record is not found.
        """
        if not event_id or not str(event_id).strip():
            raise StorageValidationError("event_id must be a non-empty string.")

        session = self._ensure_connected()
        assert self._stmt_select_event is not None

        try:
            rows = session.execute(self._stmt_select_event, (str(event_id).strip(),))
            row = rows.one()
            if row is None:
                return None

            pred_date = row.prediction_date
            if hasattr(pred_date, "date"):
                pred_date = pred_date.date()
            elif isinstance(pred_date, str):
                pred_date = datetime.date.fromisoformat(pred_date)

            return PredictionRecord(
                event_id=row.event_id,
                carrier=row.carrier,
                flight_number=row.flight_number,
                origin=row.origin,
                dest=row.dest,
                event_time=row.event_time,
                prediction_time=row.prediction_time,
                prediction_date=pred_date,
                delay_probability=float(row.delay_probability),
                predicted_delayed=bool(row.predicted_delayed),
                model_version=row.model_version,
            )
        except DriverException as exc:
            raise StorageQueryError(
                f"Failed to query prediction_by_event for event_id '{event_id}': {exc}"
            ) from exc

    def get_predictions_by_airport(
        self,
        airport: str,
        date: datetime.date | str | None = None,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> list[PredictionRecord]:
        """Retrieve recent predictions for an airport within a specific date bucket.

        Args:
            airport: 3-letter uppercase IATA airport code.
            date: Calendar date bucket. Defaults to current UTC date.
            limit: Bounded maximum number of records to return (1 <= limit <= MAX_QUERY_LIMIT).

        Returns:
            List of PredictionRecords ordered chronologically descending by prediction_time.
        """
        if not airport or not str(airport).strip():
            raise StorageValidationError("airport code must be a non-empty string.")

        airport_code = str(airport).strip().upper()
        if not IATA_CODE_REGEX.match(airport_code):
            raise StorageValidationError(f"Invalid airport IATA code: '{airport}'")

        if not (1 <= limit <= MAX_QUERY_LIMIT):
            raise StorageValidationError(
                f"Query limit must be between 1 and {MAX_QUERY_LIMIT}, got {limit}."
            )

        if date is None:
            bucket_date = datetime.datetime.now(datetime.UTC).date()
        elif isinstance(date, str):
            try:
                bucket_date = datetime.date.fromisoformat(date.strip())
            except ValueError as exc:
                raise StorageValidationError(
                    f"Invalid date format: '{date}'. Use YYYY-MM-DD."
                ) from exc
        elif isinstance(date, datetime.date):
            bucket_date = date
        else:
            raise StorageValidationError(f"Unsupported date type: {type(date).__name__}")

        session = self._ensure_connected()
        assert self._stmt_select_airport is not None

        try:
            rows = session.execute(
                self._stmt_select_airport,
                (airport_code, Date(bucket_date), int(limit)),
            )

            results: list[PredictionRecord] = []
            for row in rows:
                p_date = row.prediction_date
                if hasattr(p_date, "date"):
                    p_date = p_date.date()
                elif isinstance(p_date, str):
                    p_date = datetime.date.fromisoformat(p_date)

                results.append(
                    PredictionRecord(
                        event_id=row.event_id,
                        carrier=row.carrier,
                        flight_number=row.flight_number,
                        origin=airport_code,
                        dest=row.dest,
                        event_time=row.event_time,
                        prediction_time=row.prediction_time,
                        prediction_date=p_date,
                        delay_probability=float(row.delay_probability),
                        predicted_delayed=bool(row.predicted_delayed),
                        model_version=row.model_version,
                    )
                )
            return results
        except DriverException as exc:
            raise StorageQueryError(
                f"Failed to query predictions_by_airport for airport '{airport_code}': {exc}"
            ) from exc
