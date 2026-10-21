"""Phase 2 feature engineering and leakage-free dataset creation."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "is_delayed_15"
DELAY_THRESHOLD_MINUTES = 15.0

FEATURE_ALLOWLIST: tuple[str, ...] = (
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
)

PROHIBITED_COLUMNS: tuple[str, ...] = (
    "DEP_DELAY",
    "DEP_DEL15",
    "DEP_TIME",
    "ARR_DELAY",
    "source_file",
    "raw_record_fields",
    "raw_extra_fields",
    "_row_fingerprint",
    "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_NM",
    "DEST_CITY_NAME",
    "DEST_STATE_NM",
    "FL_DATE",
    "YEAR",
    "flight_year",
    "flight_month",
)


class FeatureEngineeringError(RuntimeError):
    """Raised when feature engineering cannot proceed."""


class TargetLeakageError(FeatureEngineeringError):
    """Raised when target leakage is detected in the model feature set."""


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for building the model feature dataset."""

    input_path: Path = Path("data/processed/accepted")
    output_path: Path = Path("data/features")
    spark_master: str = "local[*]"


def _new_spark_session(master: str) -> SparkSession:
    return (
        SparkSession.builder.master(master)
        .appName("flight-delay-feature-engineering")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )


def assert_no_leakage(columns: Sequence[str]) -> None:
    """Verify that no prohibited or outcome columns exist in the feature set.

    Distinguishes scheduled/planned fields (like DEP_TIME_BLK) from actual
    post-departure outcome fields (like DEP_TIME, DEP_DELAY, ARR_DELAY).
    """
    prohibited_set = {name.upper() for name in PROHIBITED_COLUMNS}
    for col in columns:
        normalized = col.upper()
        if normalized in prohibited_set:
            raise TargetLeakageError(f"Prohibited leakage column detected in feature set: {col}")
        # Check for actual departure/arrival time outcomes (excluding scheduled CRS/blocks)
        if normalized == "DEP_TIME" or "ACTUAL_DEP" in normalized or "ACTUAL_ARR" in normalized:
            raise TargetLeakageError(
                f"Post-departure actual time field detected in feature set: {col}"
            )
        if "DEP_DELAY" in normalized or "ARR_DELAY" in normalized:
            raise TargetLeakageError(f"Delay outcome field detected in feature set: {col}")


def _build_conditions(frame: DataFrame) -> list[tuple[str, F.Column]]:
    """Define explicit exclusion rules for missing targets and invalid features."""
    conditions: list[tuple[str, F.Column]] = []

    # 1. Target exclusion conditions
    dep_delay = F.col("DEP_DELAY")
    dep_del15 = F.col("DEP_DEL15")
    missing_target = dep_delay.isNull() | F.isnan(dep_delay) | dep_del15.isNull()
    conditions.append(("missing_target:DEP_DELAY", missing_target))

    nonfinite_delay = dep_delay.isNotNull() & (F.abs(dep_delay) == F.lit(float("inf")))
    conditions.append(("invalid_target:nonfinite_DEP_DELAY", nonfinite_delay))

    # 2. Required feature presence and domain conditions
    conditions.extend(
        [
            (
                "missing_feature:carrier",
                F.col("OP_UNIQUE_CARRIER").isNull()
                | (F.length(F.trim(F.col("OP_UNIQUE_CARRIER"))) == 0),
            ),
            (
                "missing_feature:origin",
                F.col("ORIGIN").isNull() | (F.length(F.trim(F.col("ORIGIN"))) == 0),
            ),
            (
                "missing_feature:dest",
                F.col("DEST").isNull() | (F.length(F.trim(F.col("DEST"))) == 0),
            ),
            ("missing_feature:crs_elapsed_time", F.col("CRS_ELAPSED_TIME").isNull()),
            (
                "nonpositive_feature:crs_elapsed_time",
                F.col("CRS_ELAPSED_TIME").isNotNull() & (F.col("CRS_ELAPSED_TIME") <= 0),
            ),
            ("missing_feature:distance", F.col("DISTANCE").isNull()),
            (
                "nonpositive_feature:distance",
                F.col("DISTANCE").isNotNull() & (F.col("DISTANCE") <= 0),
            ),
        ]
    )

    # Clock validity (CRS_DEP_TIME and CRS_ARR_TIME)
    for name in ("CRS_DEP_TIME", "CRS_ARR_TIME"):
        col = F.col(name)
        conditions.append(
            (
                f"invalid_clock:{name}",
                col.isNull() | (col < 0) | (col > 2400) | (F.pmod(col, F.lit(100)) >= 60),
            )
        )

    # Calendar domain checks
    month = F.col("MONTH")
    day_of_month = F.col("DAY_OF_MONTH")
    day_of_week = F.col("DAY_OF_WEEK")
    conditions.append(
        (
            "invalid_calendar_components",
            month.isNull()
            | (month < 1)
            | (month > 12)
            | day_of_month.isNull()
            | (day_of_month < 1)
            | (day_of_month > 31)
            | day_of_week.isNull()
            | (day_of_week < 1)
            | (day_of_week > 7),
        )
    )

    return conditions


def _extract_features_and_target(frame: DataFrame) -> DataFrame:
    """Transform source fields into normalized model features and binary target."""
    crs_dep = F.col("CRS_DEP_TIME")
    crs_arr = F.col("CRS_ARR_TIME")

    # Clocks: 2400 is midnight (hour 0, minute 0)
    crs_dep_hour = F.when(crs_dep == 2400, 0).otherwise(F.floor(crs_dep / 100)).cast("int")
    crs_dep_minute = F.when(crs_dep == 2400, 0).otherwise(F.pmod(crs_dep, 100)).cast("int")
    crs_arr_hour = F.when(crs_arr == 2400, 0).otherwise(F.floor(crs_arr / 100)).cast("int")
    crs_arr_minute = F.when(crs_arr == 2400, 0).otherwise(F.pmod(crs_arr, 100)).cast("int")

    # Target: 1 if DEP_DELAY >= 15.0, 0 if DEP_DELAY < 15.0
    target = F.when(F.col("DEP_DELAY") >= DELAY_THRESHOLD_MINUTES, 1).otherwise(0).cast("int")

    # Weekend indicator: Day 6 (Saturday) or Day 7 (Sunday)
    is_weekend = F.when(F.col("DAY_OF_WEEK").isin(6, 7), 1).otherwise(0).cast("int")

    return (
        frame.withColumn(TARGET_COLUMN, target)
        .withColumn("carrier", F.trim(F.col("OP_UNIQUE_CARRIER")))
        .withColumn("origin", F.trim(F.col("ORIGIN")))
        .withColumn("dest", F.trim(F.col("DEST")))
        .withColumn("crs_dep_hour", crs_dep_hour)
        .withColumn("crs_dep_minute", crs_dep_minute)
        .withColumn("dep_time_blk", F.trim(F.col("DEP_TIME_BLK")))
        .withColumn("crs_arr_hour", crs_arr_hour)
        .withColumn("crs_arr_minute", crs_arr_minute)
        .withColumn("arr_time_blk", F.trim(F.col("ARR_TIME_BLK")))
        .withColumn("month", F.col("MONTH").cast("int"))
        .withColumn("day_of_month", F.col("DAY_OF_MONTH").cast("int"))
        .withColumn("day_of_week", F.col("DAY_OF_WEEK").cast("int"))
        .withColumn("is_weekend", is_weekend)
        .withColumn("crs_elapsed_time", F.col("CRS_ELAPSED_TIME").cast("double"))
        .withColumn("distance", F.col("DISTANCE").cast("double"))
    )


def _summarize_dataset(
    annotated: DataFrame,
    conditions: list[tuple[str, F.Column]],
) -> tuple[dict[str, Any], DataFrame]:
    """Calculate reconciliation counts, target distribution, and missing feature counts."""
    metrics = [
        F.count(F.lit(1)).alias("input_rows"),
        F.sum(F.when(F.size("exclusion_reasons") == 0, 1).otherwise(0)).alias("eligible_rows"),
        F.sum(F.when(F.size("exclusion_reasons") > 0, 1).otherwise(0)).alias("excluded_rows"),
        F.sum(
            F.when(
                (F.size("exclusion_reasons") == 0) & (F.col(TARGET_COLUMN) == 1),
                1,
            ).otherwise(0)
        ).alias("target_positives"),
        F.sum(
            F.when(
                (F.size("exclusion_reasons") == 0) & (F.col(TARGET_COLUMN) == 0),
                1,
            ).otherwise(0)
        ).alias("target_negatives"),
    ]
    for name in FEATURE_ALLOWLIST:
        metrics.append(
            F.sum(
                F.when(
                    (F.size("exclusion_reasons") == 0) & F.col(name).isNull(),
                    1,
                ).otherwise(0)
            ).alias(f"missing_feature_count_{name}")
        )
    for index, (_reason, condition) in enumerate(conditions):
        metrics.append(F.sum(F.when(condition, 1).otherwise(0)).alias(f"condition_{index}"))

    row = annotated.agg(*metrics).first().asDict()
    input_rows = int(row["input_rows"] or 0)
    eligible_rows = int(row["eligible_rows"] or 0)
    excluded_rows = int(row["excluded_rows"] or 0)
    positives = int(row["target_positives"] or 0)
    negatives = int(row["target_negatives"] or 0)

    exclusion_reasons: dict[str, int] = {}
    for index, (reason, _) in enumerate(conditions):
        cnt = int(row[f"condition_{index}"] or 0)
        if cnt > 0:
            exclusion_reasons[reason] = cnt

    missing_feature_counts = {
        name: int(row[f"missing_feature_count_{name}"] or 0) for name in FEATURE_ALLOWLIST
    }

    positive_rate = round(positives / eligible_rows, 6) if eligible_rows > 0 else 0.0

    report: dict[str, Any] = {
        "status": "completed",
        "input_rows": input_rows,
        "eligible_rows": eligible_rows,
        "excluded_rows": excluded_rows,
        "final_row_count": eligible_rows,
        "counts_reconcile": input_rows == eligible_rows + excluded_rows,
        "target_distribution": {
            "target_column": TARGET_COLUMN,
            "threshold_minutes": DELAY_THRESHOLD_MINUTES,
            "positive_count (delay >= 15m)": positives,
            "negative_count (delay < 15m)": negatives,
            "positive_rate": positive_rate,
        },
        "exclusion_reasons": exclusion_reasons,
        "feature_missing_counts": missing_feature_counts,
        "feature_allowlist": list(FEATURE_ALLOWLIST),
    }

    final_dataset = annotated.where(F.size("exclusion_reasons") == 0).select(
        TARGET_COLUMN,
        *FEATURE_ALLOWLIST,
    )
    return report, final_dataset


def _write_empty_parquet(frame: DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyspark.sql.pandas.types import to_arrow_schema

    path.mkdir(parents=True, exist_ok=True)
    empty_table = pa.Table.from_batches([], schema=to_arrow_schema(frame.schema))
    pq.write_table(empty_table, str(path / "part-empty.parquet"), compression="snappy")


def _write_and_promote(
    dataset: DataFrame,
    report: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.features-", dir=output_path.parent
    ) as temp_dir:
        stage = Path(temp_dir)
        dataset_dir = stage / "dataset"
        if report["eligible_rows"] > 0:
            dataset.repartition("month").write.mode("overwrite").partitionBy("month").parquet(
                str(dataset_dir)
            )
        else:
            _write_empty_parquet(dataset, dataset_dir)

        (stage / "feature_quality_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        output_names = ("dataset", "feature_quality_report.json")
        backups: dict[str, Path] = {}
        touched: list[str] = []
        try:
            for name in output_names:
                destination = output_path / name
                if destination.exists():
                    backup = stage / f".previous-{name}"
                    if destination.is_dir():
                        shutil.copytree(destination, backup)
                    else:
                        shutil.copy2(destination, backup)
                    backups[name] = backup
                touched.append(name)
                if destination.is_dir():
                    shutil.rmtree(destination)
                elif destination.exists():
                    destination.unlink()
                source = stage / name
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
        except OSError:
            for name in reversed(touched):
                destination = output_path / name
                if destination.is_dir():
                    shutil.rmtree(destination)
                elif destination.exists():
                    destination.unlink()
            for name, backup in backups.items():
                if backup.exists():
                    if backup.is_dir():
                        shutil.copytree(backup, output_path / name)
                    else:
                        shutil.copy2(backup, output_path / name)
            raise


def build_feature_dataset(
    config: FeatureConfig | None = None,
    spark: SparkSession | None = None,
) -> dict[str, Any]:
    """Build a leakage-free ML dataset from curated accepted Parquet."""
    config = config or FeatureConfig()
    if not config.input_path.exists():
        raise FeatureEngineeringError(f"Input path does not exist: {config.input_path}")

    input_resolved = config.input_path.resolve()
    output_resolved = config.output_path.resolve()
    if input_resolved == output_resolved or output_resolved in input_resolved.parents:
        raise FeatureEngineeringError(
            f"Output path cannot be identical to or contain the input path: {config.output_path}"
        )

    owns_spark = spark is None
    session = spark or _new_spark_session(config.spark_master)
    try:
        raw_curated = session.read.parquet(str(config.input_path))
        transformed = _extract_features_and_target(raw_curated)
        conditions = _build_conditions(raw_curated)

        # Attach reason strings for each violated condition
        reason_columns = [F.when(condition, F.lit(reason)) for reason, condition in conditions]
        for index, column in enumerate(reason_columns):
            transformed = transformed.withColumn(f"_reason_{index}", column)
        annotated = transformed.withColumn(
            "exclusion_reasons",
            F.expr(
                "filter(array("
                + ", ".join(f"_reason_{index}" for index in range(len(reason_columns)))
                + "), reason -> reason is not null)"
            ),
        ).drop(*[f"_reason_{index}" for index in range(len(reason_columns))])

        annotated.persist(StorageLevel.DISK_ONLY)
        try:
            report, final_dataset = _summarize_dataset(annotated, conditions)

            # Strict leakage check on final dataset schema
            final_features = [col for col in final_dataset.columns if col != TARGET_COLUMN]
            assert_no_leakage(final_features)

            # Assert allowlist fidelity
            unexpected_features = set(final_features) - set(FEATURE_ALLOWLIST)
            if unexpected_features:
                raise FeatureEngineeringError(
                    f"Unexpected features present outside allowlist: {sorted(unexpected_features)}"
                )

            report["outputs"] = {
                "dataset_parquet": str(config.output_path / "dataset"),
                "quality_report": str(config.output_path / "feature_quality_report.json"),
            }
            LOGGER.info(
                "Feature dataset counts: input=%d eligible=%d excluded=%d "
                "(positives=%d, negatives=%d)",
                report["input_rows"],
                report["eligible_rows"],
                report["excluded_rows"],
                report["target_distribution"]["positive_count (delay >= 15m)"],
                report["target_distribution"]["negative_count (delay < 15m)"],
            )
            _write_and_promote(final_dataset, report, config.output_path)
            return report
        finally:
            annotated.unpersist()
    finally:
        if owns_spark:
            session.stop()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a leakage-free ML feature dataset from accepted flight Parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/accepted"),
        help="Input accepted Parquet directory (default: data/processed/accepted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features"),
        help="Output feature directory (default: data/features)",
    )
    parser.add_argument(
        "--master",
        default="local[*]",
        help="Spark master URL (default: local[*])",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    arguments = _argument_parser().parse_args(argv)
    try:
        report = build_feature_dataset(
            FeatureConfig(
                input_path=arguments.input,
                output_path=arguments.output,
                spark_master=arguments.master,
            )
        )
    except FeatureEngineeringError as error:
        LOGGER.error("%s", error)
        return 2
    except Exception:
        LOGGER.exception("Feature engineering failed")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
