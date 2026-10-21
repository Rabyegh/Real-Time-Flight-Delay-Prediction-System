"""Integration tests for the local Spark CSV-to-Parquet flow."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import DateType

from flight_delay.processing.historical_etl import (
    SOURCE_COLUMNS,
    ETLConfig,
    SchemaContractError,
    _new_spark_session,
    run_historical_etl,
)


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    session = _new_spark_session("local[2]")
    yield session
    session.stop()


def _valid_record(day: int = 1) -> dict[str, str]:
    return {
        "YEAR": "2023",
        "MONTH": "2",
        "DAY_OF_MONTH": str(day),
        "DAY_OF_WEEK": str(day + 2),
        "FL_DATE": f"2/{day}/2023 12:00:00 AM",
        "OP_UNIQUE_CARRIER": "ZZ",
        "ORIGIN": "AAA",
        "ORIGIN_CITY_NAME": "Alpha, ST",
        "ORIGIN_STATE_NM": "State",
        "DEST": "BBB",
        "DEST_CITY_NAME": "Beta, ST",
        "DEST_STATE_NM": "State",
        "CRS_DEP_TIME": "0741",
        "DEP_TIME": "0738",
        "DEP_DELAY": "-3.00",
        "DEP_DEL15": "0.00",
        "DEP_TIME_BLK": "0700-0759",
        "CRS_ARR_TIME": "0844",
        "ARR_DELAY": "26.00",
        "ARR_TIME_BLK": "0800-0859",
        "CRS_ELAPSED_TIME": "63.00",
        "DISTANCE": "145.00",
    }


def _write_csv(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def test_etl_parses_validates_reports_and_preserves_duplicate_rows(
    tmp_path: Path,
    spark: SparkSession,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    valid = _valid_record()
    optional_missing = _valid_record(day=2)
    for field in ("DEP_TIME", "DEP_DELAY", "DEP_DEL15", "ARR_DELAY", "CRS_ELAPSED_TIME"):
        optional_missing[field] = ""
    invalid_date = _valid_record(day=30)
    invalid_date["FL_DATE"] = "2/30/2023 12:00:00 AM"
    invalid_values = _valid_record(day=3)
    invalid_values["DEP_TIME"] = "1260"
    invalid_values["DEP_DELAY"] = "late"
    invalid_values["DEP_DEL15"] = "2.00"
    invalid_values["DAY_OF_WEEK"] = "5.5"
    date_mismatch = _valid_record(day=4)
    date_mismatch["DAY_OF_MONTH"] = "5"
    _write_csv(
        input_dir / "fixture.csv",
        [valid, optional_missing, invalid_date, invalid_values, date_mismatch, valid],
    )

    report = run_historical_etl(
        ETLConfig(input_path=input_dir, output_path=output_dir),
        spark=spark,
    )

    assert report["input_count"] == 6
    assert report["accepted_count"] == 3
    assert report["rejected_count"] == 3
    assert report["duplicate_count"] == 1
    assert report["counts_reconcile"] is True
    assert report["missing_value_counts"]["DEP_DELAY"] == 1
    assert report["missing_value_counts"]["ARR_DELAY"] == 1
    assert report["validation_results"]["invalid_date:FL_DATE"] == 1
    assert report["validation_results"]["invalid_clock:DEP_TIME"] == 1
    assert report["validation_results"]["invalid_number:DEP_DELAY"] == 1
    assert report["validation_results"]["invalid_number:DAY_OF_WEEK"] == 1
    assert report["validation_results"]["invalid_domain:DEP_DEL15"] == 1
    assert report["validation_results"]["date_components_mismatch"] == 1
    assert report["rejection_reason_counts"]["invalid_date:FL_DATE"] == 1
    assert report["rejection_reason_counts"]["invalid_clock:DEP_TIME"] == 1
    assert report["rejection_reason_counts"]["invalid_number:DEP_DELAY"] == 1
    assert report["rejection_reason_counts"]["invalid_domain:DEP_DEL15"] == 1
    assert report["rejection_reason_counts"]["date_components_mismatch"] == 1

    accepted = spark.read.parquet(str(output_dir / "accepted"))
    rejected = spark.read.parquet(str(output_dir / "rejected"))
    assert accepted.count() == 3
    assert rejected.count() == 3
    assert isinstance(accepted.schema["FL_DATE"].dataType, DateType)
    assert rejected.where("raw_DEP_DELAY = 'late'").count() == 1
    assert (output_dir / "quality_report.json").is_file()


def test_etl_rejects_rows_with_wrong_field_counts_and_preserves_raw_values(
    tmp_path: Path,
    spark: SparkSession,
) -> None:
    input_path = tmp_path / "malformed.csv"
    valid = _valid_record()
    values = [valid[name] for name in SOURCE_COLUMNS]
    extra_field = [*values, "unexpected-extra"]
    missing_trailing_field = values[:-1]
    with input_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(SOURCE_COLUMNS)
        writer.writerows([values, extra_field, missing_trailing_field])

    output_dir = tmp_path / "output"
    report = run_historical_etl(ETLConfig(input_path=input_path, output_path=output_dir), spark)
    accepted = spark.read.parquet(str(output_dir / "accepted"))
    rejected = spark.read.parquet(str(output_dir / "rejected"))

    assert report["input_count"] == 3
    assert report["accepted_count"] == accepted.count() == 1
    assert report["rejected_count"] == rejected.count() == 2
    assert report["counts_reconcile"] is True
    assert report["validation_results"]["malformed_record:too_few_fields"] == 1
    assert report["validation_results"]["malformed_record:too_many_fields"] == 1
    assert report["rejection_reason_counts"]["malformed_record:too_few_fields"] == 1
    assert report["rejection_reason_counts"]["malformed_record:too_many_fields"] == 1
    too_many = rejected.where("raw_field_count = 23").first()
    too_few = rejected.where("raw_field_count = 21").first()
    assert too_many["raw_extra_fields"] == ["unexpected-extra"]
    assert too_many["raw_record_fields"] == extra_field
    assert too_many["reject_reasons"] == ["malformed_record:too_many_fields"]
    assert too_few["raw_DISTANCE"] is None
    assert too_few["raw_record_fields"] == missing_trailing_field
    assert "malformed_record:too_few_fields" in too_few["reject_reasons"]


def test_etl_rejects_nonfinite_numeric_values_but_keeps_missing_values_null(
    tmp_path: Path,
    spark: SparkSession,
) -> None:
    input_path = tmp_path / "nonfinite.csv"
    nan_delay = _valid_record()
    nan_delay["ARR_DELAY"] = "NaN"
    infinite_distance = _valid_record()
    infinite_distance["DISTANCE"] = "Infinity"
    missing_optional = _valid_record()
    missing_optional["DEP_DELAY"] = ""
    _write_csv(input_path, [nan_delay, infinite_distance, missing_optional])

    output_dir = tmp_path / "output"
    report = run_historical_etl(ETLConfig(input_path=input_path, output_path=output_dir), spark)
    accepted = spark.read.parquet(str(output_dir / "accepted"))
    rejected = spark.read.parquet(str(output_dir / "rejected"))

    assert report["input_count"] == 3
    assert report["accepted_count"] == accepted.count() == 1
    assert report["rejected_count"] == rejected.count() == 2
    assert report["counts_reconcile"] is True
    assert report["validation_results"]["nonfinite_number:ARR_DELAY"] == 1
    assert report["validation_results"]["nonfinite_number:DISTANCE"] == 1
    assert report["rejection_reason_counts"]["nonfinite_number:ARR_DELAY"] == 1
    assert report["rejection_reason_counts"]["nonfinite_number:DISTANCE"] == 1
    assert accepted.select("DEP_DELAY").first()["DEP_DELAY"] is None
    assert rejected.where("raw_ARR_DELAY = 'NaN'").count() == 1
    assert rejected.where("raw_DISTANCE = 'Infinity'").count() == 1


def test_duplicate_reporting_uses_untrimmed_raw_field_values(
    tmp_path: Path,
    spark: SparkSession,
) -> None:
    input_path = tmp_path / "duplicates.csv"
    exact = _valid_record()
    whitespace_variant = _valid_record()
    whitespace_variant["ORIGIN"] = " AAA "
    _write_csv(input_path, [exact, exact, whitespace_variant])

    output_dir = tmp_path / "output"
    report = run_historical_etl(ETLConfig(input_path=input_path, output_path=output_dir), spark)
    accepted = spark.read.parquet(str(output_dir / "accepted"))

    assert report["input_count"] == report["accepted_count"] == accepted.count() == 3
    assert report["duplicate_count"] == 1
    assert report["counts_reconcile"] is True


def test_schema_mismatch_is_reported_without_overwriting_existing_output(
    tmp_path: Path,
    spark: SparkSession,
) -> None:
    input_path = tmp_path / "bad-schema.csv"
    fields = [field for field in SOURCE_COLUMNS if field != "DISTANCE"]
    with input_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(fields)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    marker = output_dir / "preserve-me.txt"
    marker.write_text("previous output", encoding="utf-8")

    with pytest.raises(SchemaContractError, match="missing columns: \\['DISTANCE'\\]"):
        run_historical_etl(
            ETLConfig(input_path=input_path, output_path=output_dir),
            spark=spark,
        )

    assert marker.read_text(encoding="utf-8") == "previous output"
    report = json.loads((output_dir / "quality_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["schema_issues"][0]["file"] == input_path.name
    assert report["schema_issues"][0]["issues"] == ["missing columns: ['DISTANCE']"]
