"""Integration tests for the Spark Phase 2 feature pipeline."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from flight_delay.processing.feature_engineering import (
    FEATURE_ALLOWLIST,
    PROHIBITED_COLUMNS,
    TARGET_COLUMN,
    FeatureConfig,
    FeatureEngineeringError,
    _new_spark_session,
    build_feature_dataset,
)


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    session = _new_spark_session("local[2]")
    yield session
    session.stop()


ACCEPTED_SCHEMA = StructType(
    [
        StructField("YEAR", IntegerType(), True),
        StructField("MONTH", IntegerType(), True),
        StructField("DAY_OF_MONTH", IntegerType(), True),
        StructField("DAY_OF_WEEK", IntegerType(), True),
        StructField("FL_DATE", DateType(), True),
        StructField("OP_UNIQUE_CARRIER", StringType(), True),
        StructField("ORIGIN", StringType(), True),
        StructField("ORIGIN_CITY_NAME", StringType(), True),
        StructField("ORIGIN_STATE_NM", StringType(), True),
        StructField("DEST", StringType(), True),
        StructField("DEST_CITY_NAME", StringType(), True),
        StructField("DEST_STATE_NM", StringType(), True),
        StructField("CRS_DEP_TIME", IntegerType(), True),
        StructField("DEP_TIME", IntegerType(), True),
        StructField("DEP_DELAY", DoubleType(), True),
        StructField("DEP_DEL15", IntegerType(), True),
        StructField("DEP_TIME_BLK", StringType(), True),
        StructField("CRS_ARR_TIME", IntegerType(), True),
        StructField("ARR_DELAY", DoubleType(), True),
        StructField("ARR_TIME_BLK", StringType(), True),
        StructField("CRS_ELAPSED_TIME", DoubleType(), True),
        StructField("DISTANCE", DoubleType(), True),
        StructField("source_file", StringType(), True),
        StructField("flight_year", IntegerType(), True),
        StructField("flight_month", IntegerType(), True),
    ]
)


def _row(
    dep_delay: float | None = -3.0,
    dep_del15: int | None = 0,
    day_of_week: int = 3,
    crs_dep_time: int | None = 741,
    crs_arr_time: int | None = 844,
    crs_elapsed_time: float | None = 63.0,
    distance: float | None = 145.0,
    month: int = 2,
) -> dict:
    return {
        "YEAR": 2023,
        "MONTH": month,
        "DAY_OF_MONTH": 15,
        "DAY_OF_WEEK": day_of_week,
        "FL_DATE": None,
        "OP_UNIQUE_CARRIER": "AA",
        "ORIGIN": "JFK",
        "ORIGIN_CITY_NAME": "New York, NY",
        "ORIGIN_STATE_NM": "New York",
        "DEST": "BOS",
        "DEST_CITY_NAME": "Boston, MA",
        "DEST_STATE_NM": "Massachusetts",
        "CRS_DEP_TIME": crs_dep_time,
        "DEP_TIME": 738,
        "DEP_DELAY": dep_delay,
        "DEP_DEL15": dep_del15,
        "DEP_TIME_BLK": "0700-0759",
        "CRS_ARR_TIME": crs_arr_time,
        "ARR_DELAY": -5.0,
        "ARR_TIME_BLK": "0800-0859",
        "CRS_ELAPSED_TIME": crs_elapsed_time,
        "DISTANCE": distance,
        "source_file": "feb.csv",
        "flight_year": 2023,
        "flight_month": month,
    }


def test_feature_pipeline_target_classification_and_reconciliation(
    tmp_path: Path,
    spark: SparkSession,
) -> None:
    input_dir = tmp_path / "accepted"
    output_dir = tmp_path / "features"

    records = [
        # 1. On-time: delay -3 min -> target 0
        _row(dep_delay=-3.0, dep_del15=0, day_of_week=3, crs_dep_time=741),
        # 2. Borderline on-time: delay 14.0 min -> target 0
        _row(dep_delay=14.0, dep_del15=0, day_of_week=4, crs_dep_time=800),
        # 3. Exact threshold: delay 15.0 min -> target 1
        _row(dep_delay=15.0, dep_del15=1, day_of_week=5, crs_dep_time=930),
        # 4. Delayed: delay 45.0 min -> target 1, Saturday (weekend=1)
        _row(dep_delay=45.0, dep_del15=1, day_of_week=6, crs_dep_time=1200),
        # 5. Midnight arrival: crs_arr_time=2400 -> hour 0, minute 0
        _row(dep_delay=0.0, dep_del15=0, day_of_week=7, crs_arr_time=2400),
        # 6. Midnight departure: crs_dep_time=0 -> hour 0, minute 0, accepted as midnight
        _row(dep_delay=-2.0, dep_del15=0, day_of_week=2, crs_dep_time=0),
        # 7. Missing delay outcome (cancelled/diverted/unobserved) -> excluded
        _row(dep_delay=None, dep_del15=None),
        # 8. Missing required feature: crs_elapsed_time is None -> excluded
        _row(dep_delay=10.0, dep_del15=0, crs_elapsed_time=None),
        # 9. Nonpositive feature: distance = 0.0 -> excluded
        _row(dep_delay=10.0, dep_del15=0, distance=0.0),
        # 10. Invalid clock departure: 1260 -> excluded invalid_clock:CRS_DEP_TIME
        _row(dep_delay=5.0, dep_del15=0, crs_dep_time=1260),
        # 11. Missing scheduled departure clock -> excluded missing_feature:crs_dep_time
        _row(dep_delay=5.0, dep_del15=0, crs_dep_time=None),
        # 12. Missing scheduled arrival clock -> excluded missing_feature:crs_arr_time
        _row(dep_delay=5.0, dep_del15=0, crs_arr_time=None),
        # 13. Invalid clock arrival: 2460 -> excluded invalid_clock:CRS_ARR_TIME
        _row(dep_delay=5.0, dep_del15=0, crs_arr_time=2460),
    ]

    df = spark.createDataFrame(records, schema=ACCEPTED_SCHEMA)
    df.write.partitionBy("flight_year", "flight_month").parquet(str(input_dir))

    report = build_feature_dataset(
        FeatureConfig(input_path=input_dir, output_path=output_dir),
        spark=spark,
    )

    # Reconciliation
    assert report["input_rows"] == 13
    assert report["eligible_rows"] == 6
    assert report["excluded_rows"] == 7
    assert report["counts_reconcile"] is True
    assert report["final_row_count"] == 6

    # Exclusion breakdown
    assert report["exclusion_reasons"]["missing_target:DEP_DELAY"] == 1
    assert report["exclusion_reasons"]["missing_feature:crs_elapsed_time"] == 1
    assert report["exclusion_reasons"]["nonpositive_feature:distance"] == 1
    assert report["exclusion_reasons"]["invalid_clock:CRS_DEP_TIME"] == 1
    assert report["exclusion_reasons"]["missing_feature:crs_dep_time"] == 1
    assert report["exclusion_reasons"]["missing_feature:crs_arr_time"] == 1
    assert report["exclusion_reasons"]["invalid_clock:CRS_ARR_TIME"] == 1

    # Target distribution: 2 delayed (15.0, 45.0) and 4 on-time (-3.0, 14.0, 0.0, -2.0)
    assert report["target_distribution"]["positive_count (delay >= 15m)"] == 2
    assert report["target_distribution"]["negative_count (delay < 15m)"] == 4
    assert report["target_distribution"]["positive_rate"] == 0.333333

    # Parquet dataset verification
    dataset_path = output_dir / "dataset"
    assert dataset_path.exists()
    feature_df = spark.read.parquet(str(dataset_path))
    assert feature_df.count() == 6

    # Strict schema and leakage verification
    expected_cols = {TARGET_COLUMN, *FEATURE_ALLOWLIST}
    actual_cols = set(feature_df.columns)
    assert actual_cols == expected_cols

    for prohibited in PROHIBITED_COLUMNS:
        assert prohibited not in actual_cols
        assert prohibited.lower() not in actual_cols

    # Verify extracted clock and calendar features
    midnight_arr_row = feature_df.where("crs_arr_hour = 0 AND crs_arr_minute = 0").first()
    assert midnight_arr_row is not None
    assert midnight_arr_row["is_weekend"] == 1  # Sunday

    midnight_dep_row = feature_df.where("crs_dep_hour = 0 AND crs_dep_minute = 0").first()
    assert midnight_dep_row is not None
    assert midnight_dep_row["day_of_week"] == 2  # Tuesday

    saturday_row = feature_df.where("day_of_week = 6").first()
    assert saturday_row is not None
    assert saturday_row["is_weekend"] == 1
    assert saturday_row[TARGET_COLUMN] == 1  # 45 min delay

    wednesday_row = feature_df.where("day_of_week = 3").first()
    assert wednesday_row is not None
    assert wednesday_row["is_weekend"] == 0
    assert wednesday_row["crs_dep_hour"] == 7
    assert wednesday_row["crs_dep_minute"] == 41
    assert wednesday_row[TARGET_COLUMN] == 0

    # Quality report artifact
    report_file = output_dir / "feature_quality_report.json"
    assert report_file.is_file()
    saved_report = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved_report["counts_reconcile"] is True
    assert saved_report["eligible_rows"] == 6


def test_feature_pipeline_empty_input_produces_empty_schema_parquet(
    tmp_path: Path,
    spark: SparkSession,
) -> None:
    input_dir = tmp_path / "empty_accepted"
    output_dir = tmp_path / "empty_features"

    empty_df = spark.createDataFrame([], schema=ACCEPTED_SCHEMA)
    empty_df.write.parquet(str(input_dir))

    report = build_feature_dataset(
        FeatureConfig(input_path=input_dir, output_path=output_dir),
        spark=spark,
    )

    assert report["input_rows"] == 0
    assert report["eligible_rows"] == 0
    assert report["excluded_rows"] == 0
    assert report["counts_reconcile"] is True

    # Parquet output must still be schema-bearing and readable
    dataset_df = spark.read.parquet(str(output_dir / "dataset"))
    assert dataset_df.count() == 0
    assert set(dataset_df.columns) == {TARGET_COLUMN, *FEATURE_ALLOWLIST}


def test_feature_pipeline_rejects_identical_input_and_output_path(
    tmp_path: Path,
    spark: SparkSession,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with pytest.raises(FeatureEngineeringError, match="Output path cannot be identical"):
        build_feature_dataset(
            FeatureConfig(input_path=data_dir, output_path=data_dir),
            spark=spark,
        )


def test_feature_pipeline_rejects_output_nested_inside_input(
    tmp_path: Path,
    spark: SparkSession,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    nested_dir = data_dir / "features"
    with pytest.raises(FeatureEngineeringError, match="Output path cannot be inside"):
        build_feature_dataset(
            FeatureConfig(input_path=data_dir, output_path=nested_dir),
            spark=spark,
        )
