"""Integration tests for the real-time flight delay prediction streaming pipeline.

Validates the full flow:
  Kafka / Stream source -> validation & leakage checks ->
  model inference -> Cassandra persistence,
while verifying dead-letter routing, duplicate idempotency, and error
propagation using a SQLite-backed fake that is visible to PySpark workers.

Key design note
---------------
PySpark ``local[2]`` runs partition functions in *separate Python worker
processes*.  Any callable passed to ``foreachPartition`` is serialized with
cloudpickle, then deserialized in the worker.  For deserialization to
succeed the worker must be able to ``import`` the module that defines the
callable.  We therefore:

1. Define ``SQLiteFakeCassandraStorage`` and ``StorageFactory`` in the
   separately importable ``tests.helpers.sqlite_storage`` module.
2. Add the ``tests/`` root directory to each SparkContext so that workers
   automatically have ``tests.helpers.sqlite_storage`` on their path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

# Import the picklable helpers from the standalone module
from tests.helpers.sqlite_storage import SQLiteFakeCassandraStorage, StorageFactory

from flight_delay.ingestion import generate_synthetic_flight_event
from flight_delay.processing.streaming_consumer import (
    StreamingConfig,
    StreamingPipelineError,
    build_streaming_query,
    create_prediction_batch_handler,
    reset_worker_singletons,
)
from flight_delay.storage.cassandra import CassandraConfig

# Absolute path to the tests/ root; added to SparkContext so workers can import helpers
_TESTS_ROOT = str(Path(__file__).parent.parent)


@pytest.fixture(scope="module")
def spark_session() -> SparkSession:
    """Create a local deterministic SparkSession for prediction pipeline integration tests."""
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("test-prediction-streaming-pipeline")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        # Expose the tests/ root to worker processes so they can deserialize
        # helpers.sqlite_storage.StorageFactory and SQLiteFakeCassandraStorage
        .config("spark.executorEnv.PYTHONPATH", _TESTS_ROOT)
        .config("spark.yarn.appMasterEnv.PYTHONPATH", _TESTS_ROOT)
        .getOrCreate()
    )
    # Also ensure the running driver process has tests/ on sys.path
    if _TESTS_ROOT not in sys.path:
        sys.path.insert(0, _TESTS_ROOT)
    yield spark
    spark.stop()


@pytest.fixture(autouse=True)
def clean_singletons() -> None:
    """Reset worker process singletons before and after each test."""
    reset_worker_singletons()
    yield
    reset_worker_singletons()


# -----------------------------------------------------------------------------
# 1. End-to-End Prediction Streaming Pipeline Tests
# -----------------------------------------------------------------------------


def test_prediction_pipeline_e2e_valid_event_persisted_to_cassandra(
    spark_session: SparkSession, tmp_path: Path
) -> None:
    """Verify valid events flow through validation -> model inference -> Cassandra storage."""
    source_dir = tmp_path / "source_events"
    source_dir.mkdir(parents=True, exist_ok=True)
    valid_dir = tmp_path / "out_valid"
    rejected_dir = tmp_path / "out_rejected"
    cp_dir = tmp_path / "checkpoints"
    db_path = str(tmp_path / "cassandra_fake.db")

    storage_factory = StorageFactory(db_path=db_path)
    storage = SQLiteFakeCassandraStorage(db_path=db_path)

    config = StreamingConfig(
        valid_output_dir=valid_dir,
        rejected_output_dir=rejected_dir,
        checkpoint_dir=cp_dir,
        source_format="text",
        source_path=source_dir,
        watermark_delay="10 minutes",
        trigger_interval="",
        enable_cassandra_predictions=True,
        model_path=Path("models/baseline/model.joblib"),
        model_threshold=0.5,
        storage_factory=storage_factory,
    )

    # Write 2 synthetic valid flight events
    event1 = generate_synthetic_flight_event(
        seed=201, flight_num=101, dep_hour=9, dep_min=0
    ).to_dict()
    event2 = generate_synthetic_flight_event(
        seed=202, flight_num=102, dep_hour=10, dep_min=30
    ).to_dict()
    (source_dir / "events_1.txt").write_text(
        f"{json.dumps(event1)}\n{json.dumps(event2)}\n", encoding="utf-8"
    )

    queries = build_streaming_query(spark_session, config)
    try:
        assert "predictions" in queries
        queries["valid"].processAllAvailable()
        queries["rejected"].processAllAvailable()
        queries["predictions"].processAllAvailable()
    finally:
        for q in queries.values():
            q.stop()

    # Phase 5 valid sink
    assert spark_session.read.parquet(str(valid_dir)).count() == 2

    # Both Cassandra query tables contain exactly 2 predictions
    assert storage.count_by_event() == 2
    assert storage.count_by_airport() == 2

    rec1 = storage.get_prediction_by_event(event1["event_id"])
    assert rec1 is not None
    assert rec1.event_id == event1["event_id"]
    assert rec1.carrier == event1["carrier"]
    assert rec1.flight_number == event1["flight_number"]
    assert rec1.origin == event1["origin"]
    assert rec1.dest == event1["dest"]
    assert 0.0 <= rec1.delay_probability <= 1.0
    assert rec1.predicted_delayed in (True, False)
    assert rec1.model_version == "baseline-v1"


def test_leakage_and_invalid_events_never_reach_cassandra(
    spark_session: SparkSession, tmp_path: Path
) -> None:
    """Ensure invalid and leakage-bearing events are routed to DLQ and NEVER reach Cassandra."""
    source_dir = tmp_path / "source_mixed"
    source_dir.mkdir(parents=True, exist_ok=True)
    valid_dir = tmp_path / "out_valid_mixed"
    rejected_dir = tmp_path / "out_rejected_mixed"
    cp_dir = tmp_path / "checkpoints_mixed"
    db_path = str(tmp_path / "cassandra_fake_mixed.db")

    storage_factory = StorageFactory(db_path=db_path)
    storage = SQLiteFakeCassandraStorage(db_path=db_path)

    config = StreamingConfig(
        valid_output_dir=valid_dir,
        rejected_output_dir=rejected_dir,
        checkpoint_dir=cp_dir,
        source_format="text",
        source_path=source_dir,
        watermark_delay="10 minutes",
        trigger_interval="",
        enable_cassandra_predictions=True,
        model_path=Path("models/baseline/model.joblib"),
        storage_factory=storage_factory,
    )

    valid_evt = generate_synthetic_flight_event(seed=301, flight_num=201).to_dict()

    leakage_evt = generate_synthetic_flight_event(seed=302, flight_num=202).to_dict()
    leakage_evt["dep_delay"] = 45.0

    bad_evt = generate_synthetic_flight_event(seed=303, flight_num=203).to_dict()
    bad_evt["crs_dep_hour"] = 99

    (source_dir / "mixed.txt").write_text(
        f"{json.dumps(valid_evt)}\n{json.dumps(leakage_evt)}\n{json.dumps(bad_evt)}\n",
        encoding="utf-8",
    )

    queries = build_streaming_query(spark_session, config)
    try:
        queries["valid"].processAllAvailable()
        queries["rejected"].processAllAvailable()
        queries["predictions"].processAllAvailable()
    finally:
        for q in queries.values():
            q.stop()

    assert spark_session.read.parquet(str(valid_dir)).count() == 1
    assert spark_session.read.parquet(str(rejected_dir)).count() == 2

    assert storage.count_by_event() == 1
    assert storage.get_prediction_by_event(valid_evt["event_id"]) is not None
    assert storage.get_prediction_by_event(leakage_evt["event_id"]) is None
    assert storage.get_prediction_by_event(bad_evt["event_id"]) is None


def test_replayed_event_is_idempotent_in_cassandra(
    spark_session: SparkSession, tmp_path: Path
) -> None:
    """Ensure replaying identical events produces identical prediction records in both tables."""
    source_dir = tmp_path / "source_replay"
    source_dir.mkdir(parents=True, exist_ok=True)
    valid_dir = tmp_path / "out_valid_replay"
    rejected_dir = tmp_path / "out_rejected_replay"
    cp_dir = tmp_path / "checkpoints_replay"
    db_path = str(tmp_path / "cassandra_fake_replay.db")

    storage_factory = StorageFactory(db_path=db_path)
    storage = SQLiteFakeCassandraStorage(db_path=db_path)

    config = StreamingConfig(
        valid_output_dir=valid_dir,
        rejected_output_dir=rejected_dir,
        checkpoint_dir=cp_dir,
        source_format="text",
        source_path=source_dir,
        watermark_delay="10 minutes",
        trigger_interval="",
        enable_cassandra_predictions=True,
        model_path=Path("models/baseline/model.joblib"),
        storage_factory=storage_factory,
    )

    event = generate_synthetic_flight_event(seed=401, flight_num=301).to_dict()
    (source_dir / "file1.txt").write_text(f"{json.dumps(event)}\n", encoding="utf-8")

    queries = build_streaming_query(spark_session, config)
    try:
        queries["predictions"].processAllAvailable()
        assert storage.count_by_event() == 1
        first_rec = storage.get_prediction_by_event(event["event_id"])
        assert first_rec is not None

        # Replay same event; Spark checkpoint means event1 is already committed
        # but a second identical file simulates an at-least-once duplicate
        (source_dir / "file2.txt").write_text(f"{json.dumps(event)}\n", encoding="utf-8")
        queries["predictions"].processAllAvailable()

        # Upsert semantics: both tables still have exactly 1 row with identical keys
        assert storage.count_by_event() == 1
        assert storage.count_by_airport() == 1
        second_rec = storage.get_prediction_by_event(event["event_id"])
        assert second_rec is not None
        assert second_rec.delay_probability == first_rec.delay_probability
        assert second_rec.prediction_time == first_rec.prediction_time
    finally:
        for q in queries.values():
            q.stop()


# -----------------------------------------------------------------------------
# 2. Error Handling, Missing Models, and Write Failures
# -----------------------------------------------------------------------------


def test_missing_model_artifact_raises_streaming_pipeline_error(
    spark_session: SparkSession, tmp_path: Path
) -> None:
    """Ensure a missing model path raises StreamingPipelineError early on the driver."""
    missing_path = tmp_path / "non_existent_model.joblib"

    config = StreamingConfig(
        valid_output_dir=tmp_path / "valid",
        rejected_output_dir=tmp_path / "rejected",
        checkpoint_dir=tmp_path / "checkpoints",
        source_format="text",
        source_path=tmp_path / "source",
        enable_cassandra_predictions=True,
        model_path=missing_path,
    )
    (tmp_path / "source").mkdir(parents=True, exist_ok=True)

    with pytest.raises(StreamingPipelineError, match="Model artifact not found"):
        build_streaming_query(spark_session, config)


def test_cassandra_write_failure_surfaced(spark_session: SparkSession, tmp_path: Path) -> None:
    """Ensure Cassandra write failures are raised and not silently swallowed."""
    db_path = str(tmp_path / "fail_writes.db")
    storage_factory = StorageFactory(db_path=db_path, fail_writes=True)

    event = generate_synthetic_flight_event(seed=501, flight_num=401).to_dict()
    df = spark_session.createDataFrame([event])

    from pyspark.sql.functions import col, to_timestamp

    df_staged = df.withColumn("event_time", to_timestamp(col("event_timestamp")))

    handler = create_prediction_batch_handler(
        model_path=Path("models/baseline/model.joblib"),
        cassandra_config=CassandraConfig(),
        storage_factory=storage_factory,
    )

    with pytest.raises(Exception) as exc_info:
        handler(df_staged, 1)

    assert "Simulated Cassandra node failure" in str(exc_info.value)


def test_empty_microbatch_handled_cleanly(spark_session: SparkSession, tmp_path: Path) -> None:
    """Ensure empty streaming micro-batches return immediately without database operations."""
    db_path = str(tmp_path / "empty_batch.db")
    storage_factory = StorageFactory(db_path=db_path)
    storage = SQLiteFakeCassandraStorage(db_path=db_path)

    empty_df = spark_session.createDataFrame(
        [], schema=spark_session.createDataFrame([{"a": 1}]).schema
    )

    handler = create_prediction_batch_handler(
        model_path=Path("models/baseline/model.joblib"),
        cassandra_config=CassandraConfig(),
        storage_factory=storage_factory,
    )

    handler(empty_df, 1)
    assert storage.count_by_event() == 0
    assert storage.count_by_airport() == 0


def test_microbatch_retry_produces_consistent_logical_keys_and_timestamps(
    spark_session: SparkSession, tmp_path: Path
) -> None:
    """Verify that re-executing a micro-batch produces identical prediction timestamps and keys."""
    db_path = str(tmp_path / "retry_timestamps.db")
    storage_factory = StorageFactory(db_path=db_path)
    storage = SQLiteFakeCassandraStorage(db_path=db_path)

    event = generate_synthetic_flight_event(seed=601, flight_num=501).to_dict()
    df = spark_session.createDataFrame([event])

    from pyspark.sql.functions import col, to_timestamp

    df_staged = df.withColumn("event_time", to_timestamp(col("event_timestamp")))

    handler = create_prediction_batch_handler(
        model_path=Path("models/baseline/model.joblib"),
        cassandra_config=CassandraConfig(),
        storage_factory=storage_factory,
    )

    # First micro-batch execution
    handler(df_staged, 1)
    rec1 = storage.get_prediction_by_event(event["event_id"])
    assert rec1 is not None
    initial_pred_time = rec1.prediction_time
    initial_prob = rec1.delay_probability

    # Second micro-batch execution (simulating retry)
    handler(df_staged, 1)
    assert storage.count_by_event() == 1
    assert storage.count_by_airport() == 1

    rec2 = storage.get_prediction_by_event(event["event_id"])
    assert rec2 is not None
    # Prediction time must not change on retry
    assert rec2.prediction_time == initial_pred_time
    assert rec2.delay_probability == initial_prob
