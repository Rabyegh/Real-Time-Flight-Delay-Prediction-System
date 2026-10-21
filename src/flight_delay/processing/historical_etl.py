"""Historical CSV validation and Parquet ETL for the inspected BTS extracts."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

LOGGER = logging.getLogger(__name__)

SOURCE_COLUMNS = (
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_NM",
    "DEST",
    "DEST_CITY_NAME",
    "DEST_STATE_NM",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "DEP_DEL15",
    "DEP_TIME_BLK",
    "CRS_ARR_TIME",
    "ARR_DELAY",
    "ARR_TIME_BLK",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
)
DATE_COMPONENTS = ("YEAR", "MONTH", "DAY_OF_MONTH")
REQUIRED_COLUMNS = (
    "FL_DATE",
    *DATE_COMPONENTS,
    "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
)
INTEGER_COLUMNS = ("YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK")
TIME_COLUMNS = ("CRS_DEP_TIME", "DEP_TIME", "CRS_ARR_TIME")
DOUBLE_COLUMNS = ("DEP_DELAY", "ARR_DELAY", "CRS_ELAPSED_TIME", "DISTANCE")
DATE_FORMAT = "M/d/yyyy h:mm:ss a"


class HistoricalETLError(RuntimeError):
    """Raised when an input cannot be processed under the source contract."""


class SchemaContractError(HistoricalETLError):
    """Raised when an input CSV does not match the documented source schema."""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        self.issues = issues
        details = "; ".join(f"{issue['file']}: {', '.join(issue['issues'])}" for issue in issues)
        super().__init__(f"Historical CSV schema validation failed: {details}")


@dataclass(frozen=True)
class ETLConfig:
    """Filesystem and Spark options for a historical ETL run."""

    input_path: Path = Path("data/raw")
    output_path: Path = Path("data/processed")
    spark_master: str = "local[*]"


def _source_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise HistoricalETLError(f"Input file must be a CSV: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise HistoricalETLError(f"Input path does not exist: {input_path}")
    paths = sorted(path for path in input_path.glob("*.csv") if path.is_file())
    if not paths:
        raise HistoricalETLError(f"No CSV files found in input directory: {input_path}")
    return paths


def _normalized_header(path: Path) -> tuple[list[str], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream, delimiter=","))
    except StopIteration as error:
        raise HistoricalETLError(f"CSV file is empty: {path}") from error
    except (OSError, UnicodeError, csv.Error) as error:
        raise HistoricalETLError(f"Could not read CSV header from {path}: {error}") from error

    normalized = [name.lstrip("\ufeff").strip().upper() for name in header]
    return header, normalized


def _inspect_schemas(paths: list[Path]) -> tuple[dict[Path, list[str]], list[dict[str, Any]]]:
    headers: dict[Path, list[str]] = {}
    issues: list[dict[str, Any]] = []
    expected = set(SOURCE_COLUMNS)

    for path in paths:
        _, normalized = _normalized_header(path)
        headers[path] = normalized
        reasons: list[str] = []
        duplicates = sorted(name for name, count in Counter(normalized).items() if count > 1)
        missing = sorted(expected - set(normalized))
        unexpected = sorted(set(normalized) - expected)
        if duplicates:
            reasons.append(f"duplicate columns after normalization: {duplicates}")
        if missing:
            reasons.append(f"missing columns: {missing}")
        if unexpected:
            reasons.append(f"unexpected columns: {unexpected}")
        if reasons:
            issues.append({"file": path.name, "issues": reasons})
    return headers, issues


def _new_spark_session(master: str) -> SparkSession:
    return (
        SparkSession.builder.master(master)
        .appName("flight-delay-historical-etl")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )


def _stage_csv_records(
    paths: list[Path],
    headers: dict[Path, list[str]],
    staging_path: Path,
) -> tuple[dict[Path, Path], list[Path]]:
    valid_paths: dict[Path, Path] = {}
    malformed_paths: list[Path] = []
    staging_path.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(paths):
        valid_path = staging_path / f"{index}-{path.name}"
        malformed_path = staging_path / f"{index}-{path.stem}-malformed.jsonl"
        positions = [headers[path].index(name) for name in SOURCE_COLUMNS]
        malformed_count = 0
        try:
            with (
                path.open("r", encoding="utf-8-sig", newline="") as source,
                valid_path.open("w", encoding="utf-8", newline="") as valid_stream,
                malformed_path.open("w", encoding="utf-8", newline="") as malformed_stream,
            ):
                reader = csv.reader(source, delimiter=",")
                next(reader)
                valid_writer = csv.writer(valid_stream)
                valid_writer.writerow(SOURCE_COLUMNS)
                for record in reader:
                    field_count = len(record)
                    if field_count == len(SOURCE_COLUMNS):
                        valid_writer.writerow([record[position] for position in positions])
                        continue

                    malformed_count += 1
                    malformed_record = {
                        name: record[position] if position < field_count else None
                        for name, position in zip(SOURCE_COLUMNS, positions, strict=True)
                    }
                    malformed_record.update(
                        {
                            "source_file": path.name,
                            "_raw_field_count": field_count,
                            "_raw_extra_fields": record[len(SOURCE_COLUMNS) :],
                            "_raw_record_fields": record,
                        }
                    )
                    malformed_stream.write(json.dumps(malformed_record, ensure_ascii=False) + "\n")
        except StopIteration:
            raise HistoricalETLError(f"CSV file is empty: {path}") from None
        except (OSError, UnicodeError, csv.Error) as error:
            raise HistoricalETLError(f"Could not stage CSV records from {path}: {error}") from error

        valid_paths[path] = valid_path
        if malformed_count:
            malformed_paths.append(malformed_path)
        else:
            malformed_path.unlink()

    return valid_paths, malformed_paths


def _read_csv_files(
    spark: SparkSession,
    paths: list[Path],
    valid_paths: dict[Path, Path],
    malformed_paths: list[Path],
) -> DataFrame:
    frames: list[DataFrame] = []
    for path in paths:
        schema = StructType([StructField(name, StringType(), True) for name in SOURCE_COLUMNS])
        frame = (
            spark.read.schema(schema)
            .option("header", "true")
            .option("enforceSchema", "true")
            .option("encoding", "UTF-8")
            .option("mode", "PERMISSIVE")
            .csv(str(valid_paths[path]))
            .select(*[F.col(name) for name in SOURCE_COLUMNS])
            .withColumn("source_file", F.lit(path.name))
            .withColumn("_raw_field_count", F.lit(len(SOURCE_COLUMNS)))
            .withColumn("_raw_extra_fields", F.array().cast("array<string>"))
            .withColumn("_raw_record_fields", F.array(*[F.col(name) for name in SOURCE_COLUMNS]))
        )
        frames.append(frame)

    if malformed_paths:
        malformed_schema = StructType(
            [
                *[StructField(name, StringType(), True) for name in SOURCE_COLUMNS],
                StructField("source_file", StringType(), False),
                StructField("_raw_field_count", IntegerType(), True),
                StructField(
                    "_raw_extra_fields",
                    ArrayType(StringType(), containsNull=True),
                    True,
                ),
                StructField(
                    "_raw_record_fields",
                    ArrayType(StringType(), containsNull=True),
                    True,
                ),
            ]
        )
        malformed = spark.read.schema(malformed_schema).json(
            [str(path) for path in malformed_paths]
        )
        frames.append(malformed)

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.unionByName(frame)
    return combined


def _cast_column(raw_name: str, target_type: str) -> F.Column:
    return F.expr(f"try_cast(trim(`{raw_name}`) AS {target_type})")


def _integer_column(raw_name: str) -> F.Column:
    numeric = _cast_column(raw_name, "DOUBLE")
    return F.when(
        numeric.isNotNull() & (numeric == F.floor(numeric)),
        numeric.cast("int"),
    )


def _prepare_rows(raw: DataFrame) -> tuple[DataFrame, list[tuple[str, F.Column]]]:
    original_names = {name: f"__raw__{name}" for name in SOURCE_COLUMNS}
    frame = raw.select(
        *[F.col(name).alias(original_names[name]) for name in SOURCE_COLUMNS],
        F.col("source_file"),
        F.col("_raw_field_count"),
        F.col("_raw_extra_fields"),
        F.col("_raw_record_fields"),
    )

    for name in SOURCE_COLUMNS:
        raw_value = F.col(original_names[name])
        present_value = F.when(F.length(F.trim(raw_value)) > 0, raw_value)
        if name == "FL_DATE":
            parsed = F.to_date(F.try_to_timestamp(F.trim(raw_value), F.lit(DATE_FORMAT)))
        elif name in INTEGER_COLUMNS or name in TIME_COLUMNS:
            parsed = _integer_column(original_names[name])
        elif name == "DEP_DEL15":
            numeric_flag = _cast_column(original_names[name], "DOUBLE")
            parsed = F.when(numeric_flag.isNotNull(), numeric_flag.cast("int"))
        elif name in DOUBLE_COLUMNS:
            parsed = _cast_column(original_names[name], "DOUBLE")
        else:
            parsed = present_value
        frame = frame.withColumn(name, parsed)

    conditions: list[tuple[str, F.Column]] = []
    for name in REQUIRED_COLUMNS:
        conditions.append(
            (
                f"missing_required:{name}",
                F.col(original_names[name]).isNull()
                | (F.length(F.trim(F.col(original_names[name]))) == 0),
            )
        )

    date_raw = F.col(original_names["FL_DATE"])
    conditions.append(
        (
            "invalid_date:FL_DATE",
            (F.length(F.trim(date_raw)) > 0) & F.col("FL_DATE").isNull(),
        )
    )

    numeric_columns = (*INTEGER_COLUMNS, *TIME_COLUMNS, *DOUBLE_COLUMNS, "DEP_DEL15")
    for name in numeric_columns:
        raw_value = F.col(original_names[name])
        conditions.append(
            (
                f"invalid_number:{name}",
                (F.length(F.trim(raw_value)) > 0) & F.col(name).isNull(),
            )
        )
        numeric_value = _cast_column(original_names[name], "DOUBLE")
        conditions.append(
            (
                f"nonfinite_number:{name}",
                (F.length(F.trim(raw_value)) > 0)
                & (F.isnan(numeric_value) | (F.abs(numeric_value) == F.lit(float("inf")))),
            )
        )

    conditions.extend(
        [
            (
                "malformed_record:too_few_fields",
                F.col("_raw_field_count") < len(SOURCE_COLUMNS),
            ),
            (
                "malformed_record:too_many_fields",
                F.col("_raw_field_count") > len(SOURCE_COLUMNS),
            ),
        ]
    )

    flag_raw = F.col(original_names["DEP_DEL15"])
    flag_number = _cast_column(original_names["DEP_DEL15"], "DOUBLE")
    conditions.append(
        (
            "invalid_domain:DEP_DEL15",
            (F.length(F.trim(flag_raw)) > 0)
            & flag_number.isNotNull()
            & (~flag_number.isin(0.0, 1.0)),
        )
    )

    date = F.col("FL_DATE")
    date_components_match = (
        (F.col("YEAR") == F.year(date))
        & (F.col("MONTH") == F.month(date))
        & (F.col("DAY_OF_MONTH") == F.dayofmonth(date))
    )
    complete_date = (
        date.isNotNull()
        & F.col("YEAR").isNotNull()
        & F.col("MONTH").isNotNull()
        & F.col("DAY_OF_MONTH").isNotNull()
    )
    conditions.append(("date_components_mismatch", complete_date & (~date_components_match)))
    expected_weekday = F.pmod(F.dayofweek(date) + F.lit(5), F.lit(7)) + F.lit(1)
    conditions.append(
        (
            "weekday_mismatch",
            date.isNotNull()
            & F.col("DAY_OF_WEEK").isNotNull()
            & (F.col("DAY_OF_WEEK") != expected_weekday),
        )
    )

    for name in TIME_COLUMNS:
        value = F.col(name)
        conditions.append(
            (
                f"invalid_clock:{name}",
                value.isNotNull()
                & ((value < 0) | (value > 2400) | (F.pmod(value, F.lit(100)) >= 60)),
            )
        )
    for name in ("CRS_ELAPSED_TIME", "DISTANCE"):
        conditions.append(
            (
                f"nonpositive:{name}",
                F.col(name).isNotNull() & (F.col(name) <= 0),
            )
        )

    reason_columns = [F.when(condition, F.lit(reason)) for reason, condition in conditions]
    for index, column in enumerate(reason_columns):
        frame = frame.withColumn(f"reason_{index}", column)
    frame = frame.withColumn(
        "reject_reasons",
        F.expr(
            "filter(array("
            + ", ".join(f"reason_{index}" for index in range(len(reason_columns)))
            + "), reason -> reason is not null)"
        ),
    ).drop(*[f"reason_{index}" for index in range(len(reason_columns))])

    duplicate_json = F.to_json(
        F.struct(
            F.col("_raw_field_count"),
            F.col("_raw_record_fields"),
        ),
        {"ignoreNullFields": "false"},
    )
    frame = (
        frame.withColumn("_row_fingerprint", F.sha2(duplicate_json, 256))
        .withColumn("flight_year", F.year("FL_DATE"))
        .withColumn("flight_month", F.month("FL_DATE"))
    )
    return frame, conditions


def _summarize(
    frame: DataFrame,
    conditions: list[tuple[str, F.Column]],
) -> tuple[dict[str, Any], int, dict[str, int]]:
    metrics = [
        F.count(F.lit(1)).alias("input_count"),
        F.sum(F.when(F.size("reject_reasons") == 0, 1).otherwise(0)).alias("accepted_count"),
        F.sum(F.when(F.size("reject_reasons") > 0, 1).otherwise(0)).alias("rejected_count"),
    ]
    for name in SOURCE_COLUMNS:
        metrics.append(
            F.sum(
                F.when(
                    F.col(f"__raw__{name}").isNull()
                    | (F.length(F.trim(F.col(f"__raw__{name}"))) == 0),
                    1,
                ).otherwise(0)
            ).alias(f"missing_{name}")
        )
    for index, (_reason, condition) in enumerate(conditions):
        metrics.append(F.sum(F.when(condition, 1).otherwise(0)).alias(f"validation_{index}"))

    row = frame.agg(*metrics).first().asDict()
    report: dict[str, Any] = {
        "input_count": int(row["input_count"] or 0),
        "accepted_count": int(row["accepted_count"] or 0),
        "rejected_count": int(row["rejected_count"] or 0),
        "missing_value_counts": {name: int(row[f"missing_{name}"] or 0) for name in SOURCE_COLUMNS},
        "validation_results": {
            reason: int(row[f"validation_{index}"] or 0)
            for index, (reason, _) in enumerate(conditions)
        },
    }
    duplicate_count = (
        frame.groupBy("_row_fingerprint")
        .count()
        .where(F.col("count") > 1)
        .agg(F.sum(F.col("count") - F.lit(1)).alias("duplicate_count"))
        .first()["duplicate_count"]
        or 0
    )
    reason_rows = (
        frame.select(F.explode("reject_reasons").alias("reason"))
        .groupBy("reason")
        .count()
        .orderBy("reason")
        .collect()
    )
    reason_counts = {item["reason"]: int(item["count"]) for item in reason_rows}
    return report, int(duplicate_count), reason_counts


def _write_and_promote(
    frame: DataFrame,
    report: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.etl-", dir=output_path.parent
    ) as temp_dir:
        stage = Path(temp_dir)
        accepted = frame.where(F.size("reject_reasons") == 0).select(
            *SOURCE_COLUMNS, "source_file", "flight_year", "flight_month"
        )
        rejected = frame.where(F.size("reject_reasons") > 0).select(
            *SOURCE_COLUMNS,
            *[F.col(f"__raw__{name}").alias(f"raw_{name}") for name in SOURCE_COLUMNS],
            "source_file",
            F.col("_raw_field_count").alias("raw_field_count"),
            F.col("_raw_extra_fields").alias("raw_extra_fields"),
            F.col("_raw_record_fields").alias("raw_record_fields"),
            "reject_reasons",
            "flight_year",
            "flight_month",
        )
        if report["accepted_count"]:
            accepted.repartition("flight_year", "flight_month").write.mode("overwrite").partitionBy(
                "flight_year", "flight_month"
            ).parquet(str(stage / "accepted"))
        else:
            _write_empty_parquet(accepted, stage / "accepted")
        if report["rejected_count"]:
            rejected.write.mode("overwrite").parquet(str(stage / "rejected"))
        else:
            _write_empty_parquet(rejected, stage / "rejected")
        (stage / "quality_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        output_names = ("accepted", "rejected", "quality_report.json")
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


def _write_empty_parquet(frame: DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyspark.sql.pandas.types import to_arrow_schema

    path.mkdir(parents=True, exist_ok=True)
    empty_table = pa.Table.from_batches([], schema=to_arrow_schema(frame.schema))
    pq.write_table(empty_table, str(path / "part-empty.parquet"), compression="snappy")


def _write_schema_failure_report(
    output_path: Path,
    input_files: list[Path],
    issues: list[dict[str, Any]],
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "failed",
        "input_files": [path.name for path in input_files],
        "input_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "duplicate_count": 0,
        "missing_value_counts": {},
        "schema_issues": issues,
        "validation_results": {},
        "rejection_reason_counts": {},
        "counts_reconcile": True,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path,
        prefix=".quality-report-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary_path = Path(stream.name)
    os.replace(temporary_path, output_path / "quality_report.json")


def run_historical_etl(
    config: ETLConfig | None = None,
    spark: SparkSession | None = None,
) -> dict[str, Any]:
    """Validate source CSVs, write Parquet datasets, and return a JSON-safe report."""
    config = config or ETLConfig()
    paths = _source_files(config.input_path)
    headers, schema_issues = _inspect_schemas(paths)
    if schema_issues:
        _write_schema_failure_report(config.output_path, paths, schema_issues)
        raise SchemaContractError(schema_issues)

    owns_spark = spark is None
    session = spark or _new_spark_session(config.spark_master)
    try:
        with tempfile.TemporaryDirectory(prefix="flight-delay-input-") as staging_dir:
            valid_paths, malformed_paths = _stage_csv_records(
                paths,
                headers,
                Path(staging_dir),
            )
            raw = _read_csv_files(session, paths, valid_paths, malformed_paths)
            frame, conditions = _prepare_rows(raw)
            frame.persist(StorageLevel.DISK_ONLY)
            try:
                report, duplicate_count, reason_counts = _summarize(frame, conditions)
                report.update(
                    {
                        "status": "completed",
                        "input_files": [path.name for path in paths],
                        "duplicate_count": duplicate_count,
                        "duplicate_policy": (
                            "Exact full-row field-value repeats are reported but retained; "
                            "the source has no reliable flight identifier."
                        ),
                        "rejection_reason_counts": reason_counts,
                        "schema_issues": [],
                        "outputs": {
                            "accepted_parquet": str(config.output_path / "accepted"),
                            "rejected_parquet": str(config.output_path / "rejected"),
                            "quality_report": str(config.output_path / "quality_report.json"),
                        },
                        "counts_reconcile": report["input_count"]
                        == report["accepted_count"] + report["rejected_count"],
                    }
                )
                LOGGER.info(
                    "ETL counts: input=%d accepted=%d rejected=%d exact-duplicate-candidates=%d",
                    report["input_count"],
                    report["accepted_count"],
                    report["rejected_count"],
                    duplicate_count,
                )
                _write_and_promote(frame, report, config.output_path)
                return report
            finally:
                frame.unpersist()
    finally:
        if owns_spark:
            session.stop()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate historical flight CSVs and write accepted/rejected Parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw"),
        help="CSV file or directory of CSV files (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory (default: data/processed)",
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
        report = run_historical_etl(
            ETLConfig(
                input_path=arguments.input,
                output_path=arguments.output,
                spark_master=arguments.master,
            )
        )
    except HistoricalETLError as error:
        LOGGER.error("%s", error)
        return 2
    except Exception:
        LOGGER.exception("Historical ETL failed")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
