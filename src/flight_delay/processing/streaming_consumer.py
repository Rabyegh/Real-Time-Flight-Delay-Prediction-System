"""Spark Structured Streaming consumer for real-time flight delay event ingestion.

Timestamp Semantics:
- `crs_dep_time`, `crs_dep_hour`, `crs_dep_minute`, `crs_arr_hour`, and `crs_arr_minute`
  represent local airport scheduled departure/arrival times, conforming to the historical
  BTS standard and matching the Phase 2/3 feature definition.
- `event_timestamp` and `ingestion_timestamp` represent streaming event occurrence and ingestion
  times in UTC (ISO 8601 with trailing 'Z' or '+00:00'), which are used for Spark Structured
  Streaming watermarking and stateful deduplication.

Delivery Guarantees & Duplicate Semantics:
- End-to-end processing is at-least-once: Kafka offsets and micro-batch states are tracked
  in Spark checkpoint directories.
- In-window deduplication: Duplicate `event_id` records within the configured watermark delay
  are deduplicated using stateful deduplication (`dropDuplicates(["event_id", "event_time"])`).
- File sinks write to local Parquet using Spark's atomic micro-batch commit log (`_spark_metadata`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    current_timestamp,
    date_format,
    from_json,
    struct,
    to_json,
    to_timestamp,
    udf,
    when,
)
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import BooleanType

from flight_delay.config.settings import get_settings
from flight_delay.ingestion.contract import (
    contains_prohibited_leakage,
    get_spark_flight_event_schema,
)

LOGGER = logging.getLogger(__name__)


class StreamingPipelineError(RuntimeError):
    """Base exception for streaming ingestion pipeline failures."""


class PathSafetyError(StreamingPipelineError):
    """Raised when streaming output or checkpoint directories collide or unsafely nest."""


@dataclass
class StreamingConfig:
    """Configuration options for Spark Structured Streaming consumer."""

    bootstrap_servers: str = "localhost:9092"
    input_topic: str = "flight-events"
    valid_output_dir: Path | str = "data/streaming/valid"
    rejected_output_dir: Path | str = "data/streaming/rejected"
    checkpoint_dir: Path | str = "data/streaming/checkpoints"
    watermark_delay: str = "10 minutes"
    trigger_interval: str = "5 seconds"
    publish_kafka_dlq: bool = False
    dead_letter_topic: str = "flight-events-dlq"
    spark_master: str = "local[*]"
    source_format: str = "kafka"
    source_path: Path | str | None = None
    max_offsets_per_trigger: int | None = 1000
    starting_offsets: str = "latest"


def validate_streaming_paths(
    valid_output_dir: Path | str,
    rejected_output_dir: Path | str,
    checkpoint_dir: Path | str,
) -> tuple[Path, Path, Path]:
    """Validate that streaming output sinks and checkpoint directories do not collide."""
    v_path = Path(valid_output_dir).resolve()
    r_path = Path(rejected_output_dir).resolve()
    c_path = Path(checkpoint_dir).resolve()

    paths = [("valid_output", v_path), ("rejected_output", r_path), ("checkpoint", c_path)]
    for i, (name1, p1) in enumerate(paths):
        for j, (name2, p2) in enumerate(paths):
            if i != j and p1 == p2:
                raise PathSafetyError(
                    f"Path collision: {name1} and {name2} cannot point to the same path: {p1}"
                )

    if v_path in r_path.parents or r_path in v_path.parents:
        raise PathSafetyError(
            f"Valid output ({v_path}) and rejected output ({r_path}) cannot be nested."
        )
    if c_path in v_path.parents or c_path in r_path.parents:
        raise PathSafetyError(f"Checkpoint directory ({c_path}) cannot contain output sinks.")

    return v_path, r_path, c_path


def create_streaming_spark_session(
    master: str = "local[*]",
    app_name: str = "flight-delay-streaming-consumer",
) -> SparkSession:
    """Create a configured SparkSession for streaming with Kafka and WinUtils support."""
    builder = (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5",
        )
    )
    return builder.getOrCreate()


def parse_and_validate_stream(
    raw_stream_df: DataFrame,
    watermark_delay: str = "10 minutes",
) -> tuple[DataFrame, DataFrame]:
    """Parse raw JSON events, enforce bounds, separate valid from dead-letter, and deduplicate.

    Returns:
        tuple[valid_stream_df, rejected_stream_df]
    """
    raw_json_col = col("value").cast("string").alias("raw_payload")
    parsed_col = from_json(
        raw_json_col,
        get_spark_flight_event_schema(),
        {"mode": "PERMISSIVE", "columnNameOfCorruptRecord": "_corrupt_record"},
    ).alias("parsed")

    staged_df = raw_stream_df.select(raw_json_col, parsed_col)

    # 1. Validation rule checks
    is_corrupt = col("parsed._corrupt_record").isNotNull()
    has_missing_envelope = (
        col("parsed.event_id").isNull()
        | col("parsed.event_id").rlike(r"^\s*$")
        | col("parsed.schema_version").isNull()
        | col("parsed.event_timestamp").isNull()
    )
    leakage_detector_udf = udf(contains_prohibited_leakage, BooleanType())
    has_leakage = leakage_detector_udf(col("raw_payload"))
    has_missing_fields = (
        col("parsed.carrier").isNull()
        | col("parsed.flight_number").isNull()
        | col("parsed.origin").isNull()
        | col("parsed.dest").isNull()
        | col("parsed.fl_date").isNull()
        | col("parsed.crs_dep_hour").isNull()
        | col("parsed.crs_dep_minute").isNull()
        | col("parsed.crs_arr_hour").isNull()
        | col("parsed.crs_arr_minute").isNull()
        | col("parsed.month").isNull()
        | col("parsed.day_of_month").isNull()
        | col("parsed.day_of_week").isNull()
        | col("parsed.is_weekend").isNull()
        | col("parsed.crs_elapsed_time").isNull()
        | col("parsed.distance").isNull()
    )
    has_invalid_bounds = (
        ~col("parsed.crs_dep_hour").between(0, 23)
        | ~col("parsed.crs_dep_minute").between(0, 59)
        | ~col("parsed.crs_arr_hour").between(0, 23)
        | ~col("parsed.crs_arr_minute").between(0, 59)
        | ~col("parsed.month").between(1, 12)
        | ~col("parsed.day_of_month").between(1, 31)
        | ~col("parsed.day_of_week").between(1, 7)
        | ~col("parsed.is_weekend").isin(0, 1)
        | (col("parsed.crs_elapsed_time") <= 0.0)
        | (col("parsed.distance") <= 0.0)
        | (col("parsed.origin") == col("parsed.dest"))
        | ~col("parsed.carrier").rlike("^[A-Z0-9]{2,3}$")
        | ~col("parsed.origin").rlike("^[A-Z0-9]{3}$")
        | ~col("parsed.dest").rlike("^[A-Z0-9]{3}$")
        | ~col("parsed.fl_date").rlike(r"^\d{4}-\d{2}-\d{2}$")
        | to_timestamp(col("parsed.event_timestamp")).isNull()
    )

    is_valid_condition = (
        ~is_corrupt
        & ~has_leakage
        & ~has_missing_envelope
        & ~has_missing_fields
        & ~has_invalid_bounds
    )

    rejection_code = (
        when(is_corrupt, "MALFORMED_JSON")
        .when(has_leakage, "LEAKAGE_DETECTED")
        .when(has_missing_envelope, "MISSING_ENVELOPE")
        .when(has_missing_fields, "MISSING_REQUIRED_FIELDS")
        .when(has_invalid_bounds, "OUT_OF_BOUNDS_OR_INVALID_FIELD")
        .otherwise("UNKNOWN_VALIDATION_ERROR")
    )

    rejection_reason = (
        when(is_corrupt, "Unparseable or corrupt JSON record")
        .when(has_leakage, "Prohibited post-departure leakage field found in pre-departure payload")
        .when(has_missing_envelope, "Envelope field (event_id/version/timestamp) invalid or empty")
        .when(has_missing_fields, "Required flight schedule fields missing or null")
        .when(has_invalid_bounds, "Numerical bounds or IATA/carrier pattern check failed")
        .otherwise("Event failed schema contract validation")
    )

    evaluated_df = (
        staged_df.withColumn("is_valid", is_valid_condition)
        .withColumn("error_code", when(~is_valid_condition, rejection_code).otherwise(None))
        .withColumn("rejection_reason", when(~is_valid_condition, rejection_reason).otherwise(None))
    )

    # 2. Valid Stream (watermarked and deduplicated in append mode)
    valid_candidates = evaluated_df.filter(col("is_valid")).select("parsed.*")

    # Drop internal corrupt record tracker and add event_time timestamp for watermarking
    valid_cols = [c for c in valid_candidates.columns if c != "_corrupt_record"]
    valid_with_time = (
        valid_candidates.select(valid_cols)
        .withColumn("event_time", to_timestamp(col("event_timestamp")))
        .withWatermark("event_time", watermark_delay)
        .dropDuplicates(["event_id", "event_time"])
    )

    # 3. Rejected Stream (Dead Letter Queue)
    rejected_stream = evaluated_df.filter(~col("is_valid")).select(
        col("parsed.event_id").alias("event_id"),
        col("raw_payload"),
        col("error_code"),
        col("rejection_reason"),
        date_format(current_timestamp(), "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("rejection_timestamp"),
    )

    return valid_with_time, rejected_stream


def build_streaming_query(
    spark: SparkSession,
    config: StreamingConfig,
) -> dict[str, StreamingQuery]:
    """Build and start the streaming queries for valid and dead-letter sinks."""
    v_path, r_path, c_path = validate_streaming_paths(
        config.valid_output_dir,
        config.rejected_output_dir,
        config.checkpoint_dir,
    )
    v_path.mkdir(parents=True, exist_ok=True)
    r_path.mkdir(parents=True, exist_ok=True)
    c_path.mkdir(parents=True, exist_ok=True)

    # 1. Source ingestion
    if config.source_format == "kafka":
        reader = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", config.bootstrap_servers)
            .option("subscribe", config.input_topic)
            .option("startingOffsets", config.starting_offsets)
        )
        if config.max_offsets_per_trigger:
            reader = reader.option("maxOffsetsPerTrigger", config.max_offsets_per_trigger)
        raw_stream_df = reader.load()
    elif config.source_format == "text":
        if not config.source_path:
            raise StreamingPipelineError("source_path must be provided when source_format='text'")
        raw_stream_df = spark.readStream.format("text").load(str(config.source_path))
    else:
        raise StreamingPipelineError(f"Unsupported source format: {config.source_format}")

    # 2. Processing & validation
    valid_df, rejected_df = parse_and_validate_stream(
        raw_stream_df,
        watermark_delay=config.watermark_delay,
    )

    queries: dict[str, StreamingQuery] = {}

    # 3. Valid events sink (Parquet append mode)
    valid_checkpoint = c_path / "valid"
    valid_writer = (
        valid_df.writeStream.format("parquet")
        .outputMode("append")
        .option("path", str(v_path))
        .option("checkpointLocation", str(valid_checkpoint))
    )
    if config.trigger_interval:
        valid_writer = valid_writer.trigger(processingTime=config.trigger_interval)
    valid_query = valid_writer.start()
    queries["valid"] = valid_query
    LOGGER.info("Started valid events stream query -> %s (cp: %s)", v_path, valid_checkpoint)

    # 4. Rejected events dead-letter sink (Parquet append mode)
    rejected_checkpoint = c_path / "rejected"
    rejected_writer = (
        rejected_df.writeStream.format("parquet")
        .outputMode("append")
        .option("path", str(r_path))
        .option("checkpointLocation", str(rejected_checkpoint))
    )
    if config.trigger_interval:
        rejected_writer = rejected_writer.trigger(processingTime=config.trigger_interval)
    rejected_query = rejected_writer.start()
    queries["rejected"] = rejected_query
    LOGGER.info("Started rejected events DLQ stream query -> %s", r_path)

    # 5. Optional Kafka DLQ publisher sink
    if config.publish_kafka_dlq and config.source_format == "kafka":
        dlq_kafka_checkpoint = c_path / "kafka_dlq"
        kafka_payload_df = rejected_df.select(
            col("event_id").cast("string").alias("key"),
            to_json(
                struct(
                    col("event_id"),
                    col("raw_payload"),
                    col("error_code"),
                    col("rejection_reason"),
                    col("rejection_timestamp"),
                )
            ).alias("value"),
        )
        dlq_writer = (
            kafka_payload_df.writeStream.format("kafka")
            .option("kafka.bootstrap.servers", config.bootstrap_servers)
            .option("topic", config.dead_letter_topic)
            .option("checkpointLocation", str(dlq_kafka_checkpoint))
        )
        if config.trigger_interval:
            dlq_writer = dlq_writer.trigger(processingTime=config.trigger_interval)
        dlq_kafka_query = dlq_writer.start()
        queries["kafka_dlq"] = dlq_kafka_query
        LOGGER.info(
            "Started Kafka DLQ publishing query -> %s at %s",
            config.dead_letter_topic,
            config.bootstrap_servers,
        )

    return queries


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for streaming consumer."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Run Spark Structured Streaming consumer for real-time flight delay ingestion."
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=settings.kafka_bootstrap_servers,
        help=f"Kafka bootstrap servers (default: {settings.kafka_bootstrap_servers})",
    )
    parser.add_argument(
        "--topic",
        default=settings.kafka_flight_topic,
        help=f"Input Kafka topic (default: {settings.kafka_flight_topic})",
    )
    parser.add_argument(
        "--valid-output",
        default=settings.streaming_valid_output_dir,
        help="Parquet output directory for valid events",
    )
    parser.add_argument(
        "--rejected-output",
        default=settings.streaming_rejected_output_dir,
        help="Output directory for DLQ records",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=settings.streaming_checkpoint_dir,
        help=f"Checkpoint directory (default: {settings.streaming_checkpoint_dir})",
    )
    parser.add_argument(
        "--watermark",
        default=settings.streaming_watermark_delay,
        help=f"Watermark delay duration (default: {settings.streaming_watermark_delay})",
    )
    parser.add_argument(
        "--trigger",
        default=settings.streaming_trigger_interval,
        help=f"Micro-batch trigger interval (default: {settings.streaming_trigger_interval})",
    )
    parser.add_argument(
        "--publish-dlq",
        action="store_true",
        help="Publish rejected records to Kafka DLQ topic in addition to local Parquet sink",
    )
    parser.add_argument(
        "--dlq-topic",
        default=settings.kafka_dead_letter_topic,
        help=f"Kafka dead-letter topic (default: {settings.kafka_dead_letter_topic})",
    )
    parser.add_argument(
        "--source-format",
        default="kafka",
        choices=["kafka", "text"],
        help="Source format for streaming input (default: kafka)",
    )
    parser.add_argument(
        "--source-path",
        default=None,
        help="Filesystem source path when --source-format=text",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for flight-delay-stream."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    config = StreamingConfig(
        bootstrap_servers=args.bootstrap_servers,
        input_topic=args.topic,
        valid_output_dir=args.valid_output,
        rejected_output_dir=args.rejected_output,
        checkpoint_dir=args.checkpoint_dir,
        watermark_delay=args.watermark,
        trigger_interval=args.trigger,
        publish_kafka_dlq=args.publish_dlq,
        dead_letter_topic=args.dlq_topic,
        source_format=args.source_format,
        source_path=args.source_path,
    )

    spark = create_streaming_spark_session()
    try:
        queries = build_streaming_query(spark, config)
        LOGGER.info("Active streaming queries: %s. Awaiting termination...", list(queries.keys()))
        spark.streams.awaitAnyTermination()
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Received KeyboardInterrupt; stopping streaming queries...")
        for query in spark.streams.active:
            query.stop()
        return 0
    except StreamingPipelineError as err:
        LOGGER.error("Streaming pipeline configuration failure: %s", err)
        return 2
    except Exception:
        LOGGER.exception("Unexpected error during streaming execution")
        return 1
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
