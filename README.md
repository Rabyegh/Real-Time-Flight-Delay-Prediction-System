# Real-Time Flight Delay Prediction System

A local development foundation for a flight-delay platform using US DOT Bureau of
Transportation Statistics (BTS) history and AviationStack live flight data. The
repository includes a validated historical CSV-to-Parquet ETL, leakage-free feature
engineering dataset generation, and a health-only API. Live ingestion, streaming
transformations, model training, and prediction-serving integrations are not
implemented yet.

## Architecture

```text
BTS monthly CSVs → PySpark historical ETL → accepted Parquet → PySpark feature pipeline → ML dataset + report
                                           └─ rejected Parquet

AviationStack → future ingestion → Kafka → Spark Structured Streaming
                                            ├─ MinIO (Parquet datasets)
                                            └─ Cassandra (serving state)
                                                 ↑
                                            FastAPI (health scaffold)

MLflow (SQLite backend + MinIO artifacts) · Prometheus · Grafana
Airflow is a later, optional orchestration milestone.
```

Python code uses a `src/flight_delay` package. Configuration, ingestion,
processing, storage, training, and API concerns have separate package boundaries.
The only implemented API behavior is `/health`; it does not return predictions.

## Prerequisites

- Windows, macOS, or Linux with Docker Desktop / Docker Engine and Docker Compose
  v2.24+.
- Python 3.11. Python 3.13 is not the project target: Spark 3.5.x and the declared
  interpreter range are intended for Python 3.11.
- [uv](https://docs.astral.sh/uv/) for environment and lockfile management.
- Java 17 for running Spark locally outside the provided Spark container. The
  Apache Spark image supplies its own Java runtime; a separate local Java install
  is not needed for the Compose services.
- On Windows, Spark's Hadoop local filesystem also needs Hadoop-compatible
  `winutils.exe` and `hadoop.dll` in a `HADOOP_HOME\bin` directory. Keep these
  local runtime binaries outside the repository. Linux and macOS do not need
  them.

The current Compose stack is resource-intensive. Allow several GB of memory for
Cassandra, Spark, Kafka, and monitoring services together.

## Local setup

1. Copy `.env.example` to `.env`; replace the development credentials with unique
   local values. Add your AviationStack key to `AVIATIONSTACK_API_KEY` only in
   `.env`. Never commit `.env`.
2. Install the supported interpreter if necessary, then sync the locked
   dependencies:

   ```powershell
   uv python install 3.11
   uv sync --group dev
   ```

   On systems without `uv`, install Python 3.11, create a virtual environment,
   and install the project plus development dependencies with a compatible
   package installer. `uv.lock` is the reproducible dependency resolution.
3. Check and start the local services:

   ```powershell
   docker compose config
   docker compose up --build -d
   uv run python scripts/smoke_test.py
   ```

4. Run local quality checks:

   ```powershell
   uv run ruff check .
   uv run ruff format --check .
   uv run python -m pytest
   ```

## Historical data ETL

The source files are expected under `data/raw/`; the current local inputs and
their measured schema, date ranges, null counts, and limits are documented in
[the historical data contract](docs/historical-data-contract.md). The ETL
requires the documented 22-column schema. It parses `FL_DATE` using the
observed month/day/year 12-hour timestamp format and validates date components,
weekday, numeric fields, HHmm clocks, the observed binary `DEP_DEL15` domain,
and positive non-null `CRS_ELAPSED_TIME`/`DISTANCE`.

Required fields are `FL_DATE`, `YEAR`, `MONTH`, `DAY_OF_MONTH`, `DAY_OF_WEEK`,
`OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, and `CRS_DEP_TIME`. Missing optional
measurements remain null. Malformed values and invalid records are written to
the rejected dataset with raw values and rejection reasons. Rows with fewer or
more than 22 CSV fields are rejected, with extra fields preserved in
`raw_record_fields` and `raw_extra_fields`. Non-finite numeric values such as
NaN and Infinity are rejected; delay values are not imputed or capped. There
is no flight number or reliable unique key in these files, so exact parsed
field-value duplicates are reported without trimming values, and records are
never silently deduplicated.

Run both input files, or choose one file explicitly:

```powershell
uv run flight-delay-etl --input data\raw --output data\processed
uv run flight-delay-etl --input data\raw\feb.csv --output data\processed\feb
uv run flight-delay-etl --input data\raw\june.csv --output data\processed\june
```

The `--input` option accepts a CSV file or a directory (all top-level `*.csv`
files are processed). Omitted options default to `data/raw` and `data/processed`.
The output directory contains:

```text
data/processed/
  accepted/             Accepted Parquet, partitioned by flight year and month
  rejected/             Rejected Parquet and original raw field values
  quality_report.json   Counts, missing values, validation and rejection totals
```

The report includes total input, accepted, rejected, and exact-row duplicate
candidate counts; per-column missing-value counts; schema issues; validation
counts; rejection reasons; output paths; and a count-reconciliation flag. Empty
accepted or rejected datasets are also written as readable, schema-bearing
Parquet datasets. A rerun replaces these three named outputs without deleting
other files in the configured output directory. `data/raw/` and
`data/processed/` are ignored by Git.

For local Spark on Windows, configure the runtime paths in the shell before
running the ETL (use your installed Java 17 and Hadoop-compatible binaries):

```powershell
$env:JAVA_HOME = 'C:\path\to\jdk-17'
$env:HADOOP_HOME = 'C:\path\to\hadoop'
$env:PATH = "$env:JAVA_HOME\bin;$env:HADOOP_HOME\bin;$env:PATH"
uv run flight-delay-etl --input data\raw --output data\processed
```

## Feature engineering and ML dataset (Phase 2)

Phase 2 builds a reproducible, leakage-free machine learning dataset from the
curated `accepted` Parquet data produced by the historical ETL.

### Prediction target and prediction-time assumption

- **Prediction target:** `is_delayed_15` (binary classification: `1` if departure
  delay is $\ge 15.0$ minutes, `0` if $< 15.0$ minutes).
- **Delay measure:** US DOT BTS `DEP_DELAY` in minutes. Measured against the
  observed `DEP_DEL15` flag, zero discrepancies exist across all 1,059,589
  valid rows.
- **Prediction-time assumption:** In real-time operation, predictions are made
  prior to flight departure (at or before scheduled departure time `CRS_DEP_TIME`).
  Therefore, only information planned or known before gate pushback is allowable
  as features.
- **Handling of unobserved delay outcomes:** Flights with missing departure delay
  labels (`DEP_DELAY IS NULL` or `DEP_DEL15 IS NULL`)—which include unobserved,
  cancelled, or diverted flights—are excluded from training datasets with reason
  `missing_target:DEP_DELAY`. Because cancellation indicators are absent from the
  source files, unknown outcomes are never silently assumed to be on-time (`0`).

### Feature allowlist and leakage prevention

To strictly prevent target leakage, features are restricted by an explicit
allowlist:

1. **Carrier & routing:** `carrier` (`OP_UNIQUE_CARRIER`), `origin` (`ORIGIN`),
   `dest` (`DEST`), `crs_elapsed_time` (`CRS_ELAPSED_TIME`), `distance` (`DISTANCE`).
2. **Scheduled timing:** `crs_dep_hour`, `crs_dep_minute`, `dep_time_blk`
   (`DEP_TIME_BLK`), `crs_arr_hour`, `crs_arr_minute`, `arr_time_blk` (`ARR_TIME_BLK`).
   Midnight arrivals (`2400`) are mapped to hour `0`, minute `0`.
3. **Calendar:** `month`, `day_of_month`, `day_of_week`, `is_weekend` (Saturday/Sunday).

**Strictly prohibited and excluded columns:**
- Target derivatives: `DEP_DELAY`, `DEP_DEL15`.
- Post-departure actuals: `DEP_TIME`, `ARR_DELAY`.
- High-cardinality text & redundant identifiers: `FL_DATE`, `YEAR`, `flight_year`,
  `flight_month`, city/state names (`ORIGIN_CITY_NAME`, etc.).
- Ingestion artifacts: `source_file`, `_row_fingerprint`, raw record fields.

Automated tests assert that prohibited columns never enter the final dataset.

### Generating the feature dataset

Run the feature engineering pipeline:

```powershell
uv run flight-delay-features --input data\processed\accepted --output data\features
```

The output directory contains:

```text
data/features/
  dataset/                    Feature Parquet dataset, partitioned by month
  feature_quality_report.json Reconciliation, target distribution, and missing feature counts
```

### Dataset reconciliation and measured distribution

On the complete February and June 2023 BTS dataset:

- **Input rows:** 1,080,011
- **Eligible rows (final count):** 1,059,588
- **Excluded rows:** 20,423
  - Missing departure delay target: 20,422
  - Missing `crs_elapsed_time`: 1
- **Target distribution:**
  - On-time / delay < 15m (`0`): 815,913 (77.00%)
  - Delayed $\ge 15$m (`1`): 243,675 (23.00%)
- **Feature missing counts in final dataset:** 0 across all 15 allowlisted features.
- **Count reconciliation:** $1,059,588 + 20,423 = 1,080,011$ (exact match).

### Limitations of the dataset

- **Temporal scope:** Only February and June 2023 are available. The data captures
  winter and early-summer conditions but does not represent full-year seasonality,
  fall/spring storms, or multi-year airline operations.
- **Flight identification:** The source extract has no flight number or tail number,
  preventing tail-routing or turn-around delay modeling.
- **Cancellations:** Cancellation flags are absent from the source data, so cancellations
  cannot be modeled as a distinct class.
- **Phase scope:** Phase 2 focuses exclusively on dataset curation and leakage-free
  feature engineering. Model training and offline evaluation belong to Phase 3.

To use the AviationStack API in future ingestion code, sign up for an API key,
put it in the local `.env`, and access it via `get_settings().require_aviationstack_api_key()`.
The key is optional for the health-only API scaffold.

The smoke test uploads a tiny artifact through MLflow and confirms it can be
listed again. It leaves a small run in the `infrastructure-smoke` experiment.
The optional Airflow dependency group can be installed on a supported Linux
environment with `uv sync --group dev --group orchestration`; Airflow is not part
of the default local Compose stack.

## Local service endpoints

| Service | Host endpoint | Notes |
| --- | --- | --- |
| FastAPI | <http://localhost:8000/health> | Health scaffold |
| Kafka | `localhost:9092` | Host listener; Compose clients use `kafka:29092` |
| Spark master UI | <http://localhost:8080> | Spark 3.5.5 master |
| MinIO API / console | <http://localhost:9000> / <http://localhost:9001> | Credentials from `.env`; `mlflow-artifacts` bucket initialized on startup |
| Cassandra | `localhost:9042` | CQL native transport |
| MLflow | <http://localhost:5000> | SQLite backend persisted in a named volume; artifacts use MinIO |
| Prometheus | <http://localhost:9090> | Initial self-scrape configuration |
| Grafana | <http://localhost:3000> | Prometheus datasource provisioned |

Ports can be changed in `.env`. Service-to-service URLs use Compose DNS names,
not host-published ports. Health checks report container readiness where
available; `depends_on` orders startup according to those checks but does not
make the entire platform infallible or replace runtime retry handling.

## Project layout

```text
src/flight_delay/       Application package and separated domain packages
  config/               Pydantic settings and environment loading
  ingestion/            Future BTS and AviationStack ingestion
  processing/           Historical CSV ETL; future streaming and features
  storage/              Future Cassandra and dataset persistence
  training/             Future classifier/regressor training and evaluation
  api/                  FastAPI application
tests/unit/             Configuration and API tests
tests/integration/      Synthetic Spark ETL and Parquet tests
scripts/                Local infrastructure smoke checks
configs/                Prometheus configuration
docker/                 API image and Grafana provisioning
data/                   Local-only data (subdirectories created as needed)
notebooks/              Exploratory analysis (future)
dags/                   Airflow workflows (future)
```

## Useful commands

```powershell
make install  # uv sync --group dev
make lint
make format
make test
make historical-etl
make features
make up
make logs
make smoke
make down
```

On Windows without `make`, run the corresponding `uv` and `docker compose`
commands directly.

## Troubleshooting

- **Docker daemon unavailable:** start Docker Desktop and wait for its engine to
  be ready, then rerun `docker compose config` and `docker compose up`.
- **Compose reports a missing variable:** copy `.env.example` to `.env` and
  replace every `replace-with-...` value. Credentials are deliberately required
  instead of having insecure Compose defaults.
- **Python version rejected:** install/select Python 3.11 (`uv python install
  3.11`); this project explicitly excludes Python 3.13.
- **Spark reports `JAVA_HOME` or `winutils.exe` errors:** use Java 17. On Windows,
  configure a compatible Hadoop `HADOOP_HOME` containing `bin\winutils.exe` and
  `bin\hadoop.dll`; do not put these binaries in the repository.
- **Windows application-control blocks the PySpark source build:** create the
  venv, install a temporary build backend into it, then disable build isolation
  just for PySpark:

  ```powershell
  uv venv --python 3.11
  uv pip install --python .venv\Scripts\python.exe "setuptools<81"
  uv sync --python .venv\Scripts\python.exe --group dev --locked --no-build-isolation-package pyspark
  ```

- **Kafka host vs. container address:** host applications use the published
  `localhost:9092`; Compose containers use `kafka:29092`.
- **A service is still starting:** inspect `docker compose ps` and
  `docker compose logs <service>`. Cassandra and the first MinIO/Kafka startup
  can take longer than the API.

## Next milestone

Phase 3: Model training and offline evaluation. Establish baseline classifiers
(e.g., Logistic Regression, LightGBM/XGBoost) using the curated feature dataset,
implement time-based train/validation splitting, log experiments to MLflow, and
evaluate PR-AUC, ROC-AUC, and class-imbalanced metrics.
