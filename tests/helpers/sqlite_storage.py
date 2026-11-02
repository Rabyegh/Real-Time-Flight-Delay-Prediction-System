"""SQLite-backed fake CassandraStorage for PySpark integration tests.

This module is intentionally separate from the test files so that PySpark
worker processes can import it directly.  Workers running in local[2] mode
run in separate Python sub-processes and therefore cannot import classes
that are only defined inside a pytest test module.

Usage
-----
In a test fixture create the shared SQLite database, then pass a
``StorageFactory`` instance as ``storage_factory`` on ``StreamingConfig``::

    db_path = str(tmp_path / "preds.db")
    factory = StorageFactory(db_path)
    config = StreamingConfig(
        ...
        storage_factory=factory,
    )
"""

from __future__ import annotations

import datetime
import sqlite3

from flight_delay.storage.cassandra import PredictionRecord, StorageWriteError

_DDL_BY_EVENT = """
CREATE TABLE IF NOT EXISTS predictions_by_event (
    event_id        TEXT PRIMARY KEY,
    carrier         TEXT,
    flight_number   TEXT,
    origin          TEXT,
    dest            TEXT,
    event_time      TEXT,
    prediction_time TEXT,
    prediction_date TEXT,
    delay_probability REAL,
    predicted_delayed INTEGER,
    model_version   TEXT
)
"""

_DDL_BY_AIRPORT = """
CREATE TABLE IF NOT EXISTS predictions_by_airport (
    airport         TEXT,
    prediction_date TEXT,
    prediction_time TEXT,
    event_id        TEXT,
    carrier         TEXT,
    flight_number   TEXT,
    dest            TEXT,
    event_time      TEXT,
    delay_probability REAL,
    predicted_delayed INTEGER,
    model_version   TEXT,
    PRIMARY KEY (airport, prediction_date, prediction_time, event_id)
)
"""


class SQLiteFakeCassandraStorage:
    """SQLite-backed replacement for CassandraStorage usable across worker processes.

    All reads/writes go through the SQLite file, which is shared between the
    driver process and every executor process, making partition-level writes
    visible from the driver after ``processAllAvailable()``.
    """

    def __init__(self, db_path: str, fail_writes: bool = False) -> None:
        self.db_path = str(db_path)
        self.fail_writes = fail_writes
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_DDL_BY_EVENT)
            conn.execute(_DDL_BY_AIRPORT)
            conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:  # noqa: D401
        pass

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def upsert_prediction_by_event(self, record: PredictionRecord) -> None:
        if self.fail_writes:
            raise StorageWriteError("Simulated Cassandra node failure during event upsert")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO predictions_by_event VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.event_id,
                    record.carrier,
                    record.flight_number,
                    record.origin,
                    record.dest,
                    record.event_time.isoformat(),
                    record.prediction_time.isoformat(),
                    str(record.prediction_date),
                    record.delay_probability,
                    int(record.predicted_delayed),
                    record.model_version,
                ),
            )
            conn.commit()

    def upsert_prediction_by_airport(self, record: PredictionRecord) -> None:
        if self.fail_writes:
            raise StorageWriteError("Simulated Cassandra node failure during airport upsert")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO predictions_by_airport VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.origin,
                    str(record.prediction_date),
                    record.prediction_time.isoformat(),
                    record.event_id,
                    record.carrier,
                    record.flight_number,
                    record.dest,
                    record.event_time.isoformat(),
                    record.delay_probability,
                    int(record.predicted_delayed),
                    record.model_version,
                ),
            )
            conn.commit()

    def upsert_prediction(self, record: PredictionRecord) -> None:
        self.upsert_prediction_by_event(record)
        self.upsert_prediction_by_airport(record)

    def upsert_predictions_batch(self, records: list[PredictionRecord]) -> None:
        for idx, rec in enumerate(records):
            try:
                self.upsert_prediction(rec)
            except Exception as exc:
                raise StorageWriteError(
                    f"Failed upserting prediction for event '{rec.event_id}' "
                    f"at batch index {idx}: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Reads (driver-side verification helpers)
    # ------------------------------------------------------------------
    def get_prediction_by_event(self, event_id: str) -> PredictionRecord | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM predictions_by_event WHERE event_id = ?", (event_id,)
            ).fetchone()
        if not row:
            return None
        return PredictionRecord(
            event_id=row[0],
            carrier=row[1],
            flight_number=row[2],
            origin=row[3],
            dest=row[4],
            event_time=datetime.datetime.fromisoformat(row[5]),
            prediction_time=datetime.datetime.fromisoformat(row[6]),
            prediction_date=datetime.date.fromisoformat(row[7]),
            delay_probability=row[8],
            predicted_delayed=bool(row[9]),
            model_version=row[10],
        )

    def count_by_event(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions_by_event").fetchone()[0]

    def count_by_airport(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions_by_airport").fetchone()[0]


class StorageFactory:
    """Picklable callable that creates a SQLiteFakeCassandraStorage per worker process.

    Using a class with ``__call__`` (rather than a lambda or nested function)
    allows cloudpickle to serialize it by reference to the *module* rather
    than by value, so PySpark worker processes only need to be able to
    ``import tests.helpers.sqlite_storage`` — which works as long as the
    ``tests/`` directory is on ``sys.path``.
    """

    def __init__(self, db_path: str, fail_writes: bool = False) -> None:
        self.db_path = str(db_path)
        self.fail_writes = fail_writes

    def __call__(self) -> SQLiteFakeCassandraStorage:
        return SQLiteFakeCassandraStorage(db_path=self.db_path, fail_writes=self.fail_writes)
