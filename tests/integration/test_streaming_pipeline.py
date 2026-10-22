"""Integration tests for PySpark Structured Streaming consumer, deduplication, and DLQ sinks."""

import json
import sys
from pathlib import Path

import pytest

from flight_delay.ingestion import generate_synthetic_flight_event
from flight_delay.ingestion.contract import PROHIBITED_LEAKAGE_COLUMNS
from flight_delay.processing.streaming_consumer import (
    PathSafetyError,
    StreamingConfig,
    build_streaming_query,
    parse_and_validate_stream,
    validate_streaming_paths,
)


@pytest.fixture(scope="module")
def spark_session():
    """Create a local, deterministic SparkSession for streaming pipeline tests."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("test-streaming-pipeline")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_validate_streaming_paths_safety(tmp_path: Path) -> None:
    valid_dir = tmp_path / "valid"
    rejected_dir = tmp_path / "rejected"
    checkpoint_dir = tmp_path / "checkpoint"

    # Valid separate paths
    v, r, c = validate_streaming_paths(valid_dir, rejected_dir, checkpoint_dir)
    assert v == valid_dir.resolve()
    assert r == rejected_dir.resolve()
    assert c == checkpoint_dir.resolve()

    # Identical paths collision
    with pytest.raises(PathSafetyError, match="cannot point to the same path"):
        validate_streaming_paths(valid_dir, valid_dir, checkpoint_dir)

    # Nested output paths
    nested_rejected = valid_dir / "nested_rejected"
    with pytest.raises(PathSafetyError, match="cannot be nested"):
        validate_streaming_paths(valid_dir, nested_rejected, checkpoint_dir)

    # Checkpoint containing output sink
    with pytest.raises(PathSafetyError, match="cannot contain output sinks"):
        validate_streaming_paths(checkpoint_dir / "valid", rejected_dir, checkpoint_dir)


def test_parse_and_validate_stream_routing_and_dlq(spark_session, tmp_path: Path) -> None:
    # 1. Prepare sample records
    valid1 = generate_synthetic_flight_event(seed=10, flight_num=101).to_dict()
    valid2 = generate_synthetic_flight_event(seed=20, flight_num=202).to_dict()

    corrupt_json = '{"event_id": "bad", "carrier": "AA", unclosed_json'
    missing_fields = {"event_id": "missing_fields", "schema_version": "1.0.0"}
    invalid_bounds = generate_synthetic_flight_event(seed=30).to_dict()
    invalid_bounds["crs_dep_hour"] = 99
    leakage_event = generate_synthetic_flight_event(seed=40).to_dict()
    leakage_event["dep_delay"] = 25.0

    records = [
        json.dumps(valid1),
        json.dumps(valid2),
        corrupt_json,
        json.dumps(missing_fields),
        json.dumps(invalid_bounds),
        json.dumps(leakage_event),
    ]

    source_dir = tmp_path / "raw_text_stream"
    source_dir.mkdir(parents=True, exist_ok=True)
    input_file = source_dir / "batch_001.txt"
    input_file.write_text("\n".join(records) + "\n", encoding="utf-8")

    # 2. Read as streaming DataFrame
    raw_stream_df = spark_session.readStream.format("text").load(str(source_dir))

    valid_df, rejected_df = parse_and_validate_stream(
        raw_stream_df,
        watermark_delay="10 minutes",
    )

    valid_sink_dir = tmp_path / "valid_sink"
    valid_cp = tmp_path / "valid_cp"
    rejected_sink_dir = tmp_path / "rejected_sink"
    rejected_cp = tmp_path / "rejected_cp"

    valid_q = (
        valid_df.writeStream.format("parquet")
        .outputMode("append")
        .option("path", str(valid_sink_dir))
        .option("checkpointLocation", str(valid_cp))
        .start()
    )

    rejected_q = (
        rejected_df.writeStream.format("parquet")
        .outputMode("append")
        .option("path", str(rejected_sink_dir))
        .option("checkpointLocation", str(rejected_cp))
        .start()
    )

    try:
        valid_q.processAllAvailable()
        rejected_q.processAllAvailable()
    finally:
        valid_q.stop()
        rejected_q.stop()

    # 3. Assert on output Parquet sinks
    valid_result_df = spark_session.read.parquet(str(valid_sink_dir))
    rejected_result_df = spark_session.read.parquet(str(rejected_sink_dir))

    assert valid_result_df.count() == 2
    valid_event_ids = {row["event_id"] for row in valid_result_df.collect()}
    assert valid1["event_id"] in valid_event_ids
    assert valid2["event_id"] in valid_event_ids

    # Verify all 15 allowlisted feature columns exist in valid sink
    for feat in (
        "carrier",
        "origin",
        "dest",
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
    ):
        assert feat in valid_result_df.columns

    # Verify rejected DLQ sink
    assert rejected_result_df.count() == 4
    error_codes = {row["error_code"] for row in rejected_result_df.collect()}
    assert "MALFORMED_JSON" in error_codes
    assert "MISSING_REQUIRED_FIELDS" in error_codes or "MISSING_ENVELOPE" in error_codes
    assert "OUT_OF_BOUNDS_OR_INVALID_FIELD" in error_codes
    assert "LEAKAGE_DETECTED" in error_codes


def test_streaming_deduplication_drops_duplicate_events(spark_session, tmp_path: Path) -> None:
    event = generate_synthetic_flight_event(seed=99).to_dict()
    event_json = json.dumps(event)

    # 3 identical copies of the exact same event
    records = [event_json, event_json, event_json]

    source_dir = tmp_path / "dedup_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    input_file = source_dir / "batch_dedup.txt"
    input_file.write_text("\n".join(records) + "\n", encoding="utf-8")

    raw_stream_df = spark_session.readStream.format("text").load(str(source_dir))
    valid_df, _ = parse_and_validate_stream(raw_stream_df, watermark_delay="10 minutes")

    valid_sink_dir = tmp_path / "dedup_valid_sink"
    valid_cp = tmp_path / "dedup_valid_cp"

    query = (
        valid_df.writeStream.format("parquet")
        .outputMode("append")
        .option("path", str(valid_sink_dir))
        .option("checkpointLocation", str(valid_cp))
        .start()
    )

    try:
        query.processAllAvailable()
    finally:
        query.stop()

    result_df = spark_session.read.parquet(str(valid_sink_dir))
    # Deduplication inside watermark must yield exactly 1 record
    assert result_df.count() == 1
    assert result_df.first()["event_id"] == event["event_id"]


def test_build_streaming_query_checkpoint_recovery(spark_session, tmp_path: Path) -> None:
    source_dir = tmp_path / "source_batches"
    source_dir.mkdir(parents=True, exist_ok=True)
    valid_dir = tmp_path / "recovered_valid"
    rejected_dir = tmp_path / "recovered_rejected"
    cp_dir = tmp_path / "recovered_cp"

    config = StreamingConfig(
        valid_output_dir=valid_dir,
        rejected_output_dir=rejected_dir,
        checkpoint_dir=cp_dir,
        source_format="text",
        source_path=source_dir,
        watermark_delay="10 minutes",
        trigger_interval="",
    )

    # Batch 1: write 2 valid events in chronological order (08:00, 08:15)
    e1 = generate_synthetic_flight_event(seed=101, flight_num=1, dep_hour=8, dep_min=0).to_dict()
    e2 = generate_synthetic_flight_event(seed=102, flight_num=2, dep_hour=8, dep_min=15).to_dict()
    (source_dir / "file1.txt").write_text(f"{json.dumps(e1)}\n{json.dumps(e2)}\n", encoding="utf-8")

    queries1 = build_streaming_query(spark_session, config)
    try:
        queries1["valid"].processAllAvailable()
        queries1["rejected"].processAllAvailable()
    finally:
        for q in queries1.values():
            q.stop()

    assert spark_session.read.parquet(str(valid_dir)).count() == 2

    # Batch 2: add a 3rd valid event (08:30) with event_time strictly after watermark
    import time

    time.sleep(1.5)
    e3 = generate_synthetic_flight_event(seed=103, flight_num=3, dep_hour=8, dep_min=30).to_dict()
    (source_dir / "file2.txt").write_text(f"{json.dumps(e3)}\n", encoding="utf-8")

    # Restart queries from existing checkpoint
    queries2 = build_streaming_query(spark_session, config)
    try:
        queries2["valid"].processAllAvailable()
        queries2["rejected"].processAllAvailable()
    finally:
        for q in queries2.values():
            q.stop()

    # Total must now be exactly 3 without duplicating batch 1
    recovered_df = spark_session.read.parquet(str(valid_dir))
    assert recovered_df.count() == 3
    event_ids = {row["event_id"] for row in recovered_df.collect()}
    assert {e1["event_id"], e2["event_id"], e3["event_id"]} == event_ids


def test_streaming_rejects_every_prohibited_leakage_column(spark_session, tmp_path: Path) -> None:
    """Ensure every prohibited leakage column is rejected to DLQ and none enters valid sink."""
    source_dir = tmp_path / "leakage_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    valid_dir = tmp_path / "leakage_valid"
    rejected_dir = tmp_path / "leakage_rejected"
    cp_dir = tmp_path / "leakage_cp"

    clean_event = generate_synthetic_flight_event(seed=500, flight_num=500).to_dict()
    records = [json.dumps(clean_event)]

    for idx, col_name in enumerate(sorted(PROHIBITED_LEAKAGE_COLUMNS)):
        seed_val = 600 + idx
        leakage_event = generate_synthetic_flight_event(
            seed=seed_val, flight_num=seed_val
        ).to_dict()
        leakage_event[col_name] = 15.0 if "delay" in col_name else "1430"
        records.append(json.dumps(leakage_event))

    (source_dir / "batch.txt").write_text("\n".join(records) + "\n", encoding="utf-8")

    config = StreamingConfig(
        valid_output_dir=valid_dir,
        rejected_output_dir=rejected_dir,
        checkpoint_dir=cp_dir,
        source_format="text",
        source_path=source_dir,
        watermark_delay="10 minutes",
        trigger_interval="",
    )

    queries = build_streaming_query(spark_session, config)
    try:
        queries["valid"].processAllAvailable()
        queries["rejected"].processAllAvailable()
    finally:
        for q in queries.values():
            q.stop()

    valid_df = spark_session.read.parquet(str(valid_dir))
    rejected_df = spark_session.read.parquet(str(rejected_dir))

    assert valid_df.count() == 1
    assert valid_df.first()["event_id"] == clean_event["event_id"]

    assert rejected_df.count() == len(PROHIBITED_LEAKAGE_COLUMNS)
    error_codes = {row["error_code"] for row in rejected_df.collect()}
    assert error_codes == {"LEAKAGE_DETECTED"}


def test_streaming_rejects_null_empty_and_whitespace_event_ids(
    spark_session, tmp_path: Path
) -> None:
    """Ensure null, empty, and whitespace-only event_ids are rejected to DLQ."""
    source_dir = tmp_path / "event_id_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    valid_dir = tmp_path / "event_id_valid"
    rejected_dir = tmp_path / "event_id_rejected"
    cp_dir = tmp_path / "event_id_cp"

    clean_event = generate_synthetic_flight_event(seed=700, flight_num=700).to_dict()
    records = [json.dumps(clean_event)]

    null_id_event = generate_synthetic_flight_event(seed=701, flight_num=701).to_dict()
    null_id_event["event_id"] = None
    records.append(json.dumps(null_id_event))

    empty_id_event = generate_synthetic_flight_event(seed=702, flight_num=702).to_dict()
    empty_id_event["event_id"] = ""
    records.append(json.dumps(empty_id_event))

    ws_id_event = generate_synthetic_flight_event(seed=703, flight_num=703).to_dict()
    ws_id_event["event_id"] = "   \t  "
    records.append(json.dumps(ws_id_event))

    (source_dir / "batch.txt").write_text("\n".join(records) + "\n", encoding="utf-8")

    config = StreamingConfig(
        valid_output_dir=valid_dir,
        rejected_output_dir=rejected_dir,
        checkpoint_dir=cp_dir,
        source_format="text",
        source_path=source_dir,
        watermark_delay="10 minutes",
        trigger_interval="",
    )

    queries = build_streaming_query(spark_session, config)
    try:
        queries["valid"].processAllAvailable()
        queries["rejected"].processAllAvailable()
    finally:
        for q in queries.values():
            q.stop()

    valid_df = spark_session.read.parquet(str(valid_dir))
    rejected_df = spark_session.read.parquet(str(rejected_dir))

    assert valid_df.count() == 1
    assert valid_df.first()["event_id"] == clean_event["event_id"]

    assert rejected_df.count() == 3
    for row in rejected_df.collect():
        assert row["error_code"] == "MISSING_ENVELOPE"
