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
  processing/           Historical CSV ETL and leakage-free feature engineering
  storage/              Future Cassandra and dataset persistence
  training/             Baseline classifier training, evaluation, and artifact generation
  api/                  FastAPI application
tests/unit/             Unit tests (settings, health, feature engineering, baseline training)
tests/integration/      Integration tests (PySpark ETL, feature pipeline, baseline pipeline)
scripts/                Local infrastructure smoke checks
configs/                Prometheus configuration
docker/                 API image and Grafana provisioning
data/                   Local-only data (raw, processed, features, samples, models)
models/                 Git-ignored model artifacts (models/baseline/)
notebooks/              Exploratory analysis (future)
dags/                   Airflow workflows (future)
```

## Baseline Model Training and Evaluation (Phase 3)

The baseline model training pipeline establishes reproducible, leakage-free reference
benchmarks on the curated historical feature dataset (`data/features/dataset`).

### Prediction Target & Horizon

- **Target:** `is_delayed_15` (binary classification: `1` if `DEP_DELAY >= 15.0` minutes, `0` otherwise).
- **Prediction Horizon:** Planned pre-departure time. The model uses only features known before
  the flight departs. All post-departure actual timestamps (`DEP_TIME`, `ARR_TIME`), delays
  (`DEP_DELAY`, `ARR_DELAY`), and metadata are strictly excluded.

### Feature Allowlist

15 pre-departure features:
- **Categoricals (5):** `carrier`, `origin`, `dest`, `dep_time_blk`, `arr_time_blk` (one-hot encoded,
  handling unknown categories safely).
- **Numericals (10):** `crs_dep_hour`, `crs_dep_minute`, `crs_arr_hour`, `crs_arr_minute`, `month`,
  `day_of_month`, `day_of_week`, `is_weekend`, `crs_elapsed_time`, `distance` (standard scaled).

### Chronological Data Split Strategy

Because the available BTS data covers only February and June 2023 (with a 3-month gap between them),
a strict chronological boundary split is used to evaluate future flights from past data while
preventing day-of-week distortion:

| Subset | Date Range | Row Count | Delay Rate | Rationale |
|---|---|---|---|---|
| **Train** | `2023-02-01` to `2023-06-16` | 798,992 (75.4%) | 20.58% | All February (winter) + June 1-16 (early summer) |
| **Validation** | `2023-06-17` to `2023-06-23` | 132,728 (12.5%) | 28.56% | Exactly 1 full 7-day cycle (Sat to Fri) in mid-June |
| **Test** | `2023-06-24` to `2023-06-30` | 127,868 (12.1%) | 32.30% | Exactly 1 full 7-day cycle (Sat to Fri) in late June |

- **Date Disjointness:** Strictly enforced. No calendar date appears in multiple subsets.
- **Leakage Prevention:** All preprocessing transformers (`OneHotEncoder`, `StandardScaler`) are
  fitted exclusively on `X_train`. The test set remains completely isolated until final evaluation.
- **Validation Role:** The validation partition (`2023-06-17` to `2023-06-23`) provides an
  intermediate temporal benchmark to monitor seasonal progression. It is reported for comparison
  and reserved for future hyperparameter sweeps and threshold tuning; no tuning was applied in this baseline phase.

### Evaluation Results Across Partitions

| Metric | Majority Baseline (Test) | Logistic Regression (Validation) | Logistic Regression (Test) | Test vs Baseline Delta |
|---|---|---|---|---|
| **Accuracy** | 67.70% | 73.41% | 69.10% | +1.40% |
| **Precision** | 0.00% | 61.09% | 54.05% | +54.05% |
| **Recall** | 0.00% | 19.01% | 28.89% | +28.89% |
| **F1-Score** | 0.0000 | 0.2900 | 0.3766 | +0.3766 |
| **ROC-AUC** | 0.5000 | 0.7158 | 0.6840 | +0.1840 |
| **PR-AUC (Avg Precision)** | 0.3230 | 0.4941 | 0.4888 | +0.1658 |
| **Prevalence** | 32.30% | 28.56% | 32.30% | — |

**Confusion Matrix (Held-Out Test Set, Late June, N = 127,868):**
- True Negatives: 76,417 | False Positives: 10,146
- False Negatives: 29,371 | True Positives: 11,934

**Confusion Matrix (Validation Set, Mid-June, N = 132,728):**
- True Negatives: 90,226 | False Positives: 4,591
- False Negatives: 30,703 | True Positives: 7,208

### Why Accuracy Alone Is Insufficient

In an imbalanced setting (32.3% delay rate in late June):
- The naive majority baseline scores **67.7% accuracy** simply by predicting that every single flight
  is on-time. Its precision, recall, and F1 are **0.0000**, and its ROC-AUC is **0.5000** (random).
- Measuring accuracy alone rewards a model that fails to detect any delays.
- Logistic Regression achieves an ROC-AUC of **0.6840** and PR-AUC of **0.4888** (significantly above
  the 0.3230 baseline), correctly catching 11,934 delayed flights with 54.05% precision.

### How to Run Baseline Training

```powershell
# Using Makefile
make train

# Or via CLI
uv run flight-delay-train --input data/features/dataset --output models/baseline
```

Generated artifacts (stored under Git-ignored `models/baseline/`):
- `model.joblib`: Serialized scikit-learn Logistic Regression pipeline.
- `dummy_model.joblib`: Serialized majority-class baseline.
- `evaluation_metrics.json`: Full machine-readable metrics across train, val, and test.
- `split_metadata.json`: Date ranges, sample counts, and class distributions.
- `evaluation_report.md`: Structured evaluation report with markdown tables.

### Limitations & Non-Production Notice

> [!WARNING]
> This model is a **baseline reference benchmark** and is **NOT production-ready**.

1. **Non-Consecutive Coverage:** Only February and June 2023 data are available, leaving March,
   April, and May unobserved.
2. **Seasonal Drift:** Summer travel congestion drives delay rates from 18.5% in February to
   32.3% in late June. A model trained primarily on winter data faces significant concept drift.
3. **No Generalization Guarantee:** Results from two months in 2023 do not establish generalization
   to autumn, holiday peaks, other operating years, or airlines not represented in the sample.
4. **Pre-Departure Horizon:** Static schedule features alone cannot capture real-time weather
   disruptions, gate holds, or upstream aircraft turnaround delays.

## Useful commands

```powershell
make install  # uv sync --group dev
make lint
make format
make test
make historical-etl
make features
make train
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

## MLflow Experiment Tracking (Phase 4)

MLflow experiment tracking is integrated into the baseline model training pipeline to record
reproducible hyperparameters, separated temporal evaluation metrics, model signatures, and
training artifacts.

### Configuration & CLI Options

The training pipeline supports flexible MLflow configuration via CLI flags, environment variables,
or `Makefile` targets:

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| Tracking URI | `--tracking-uri` | `file:./mlruns` or `$MLFLOW_TRACKING_URI` | MLflow tracking backend URI (local directory or remote store) |
| Experiment Name | `--experiment-name` | `flight-delay-baseline` | Name of the MLflow experiment |
| Run Name | `--run-name` | `logistic-regression-baseline-YYYYMMDD_HHMMSS` | Descriptive identifier for the run |
| Disable Tracking | `--no-mlflow` | False | Disables MLflow tracking when running offline or in unit tests |

### Running Training with MLflow Tracking

```powershell
# Using default local file store (./mlruns) and default experiment via Makefile
make train

# Or specify custom tracking URI and experiment name via CLI
uv run flight-delay-train `
  --input data/features/dataset `
  --output models/baseline `
  --tracking-uri file:./mlruns `
  --experiment-name flight-delay-baseline `
  --run-name logistic-regression-baseline-run

# Run with MLflow disabled
uv run flight-delay-train --no-mlflow
```

### Logged Run Information

Each MLflow run logs:
- **Tags:** `model_family="baseline"`, `algorithm="logistic_regression"`, `target_column="is_delayed_15"`,
  `split_strategy="chronological_date_disjoint"`, `validation_role="temporal_monitoring_and_tuning_reserve"`,
  `test_role="isolated_final_evaluation"`, `production_ready="false"`.
- **Parameters:** Model hyperparameters (`max_iter`, `solver`, `class_weight`, `random_state`,
  `decision_threshold`), dataset split dates (`train_dates`, `val_dates`, `test_dates`), sample counts
  (`train_rows`, `val_rows`, `test_rows`, `total_rows`), and feature lists (`feature_count=15`, categorical
  and numerical feature sets).
- **Metrics:** Train, validation, and test metrics are logged with explicit partition prefixes:
  - **Train:** `train_accuracy`, `train_precision`, `train_recall`, `train_f1_score`, `train_roc_auc`,
    `train_pr_auc`, `train_positive_prevalence`.
  - **Validation:** `val_accuracy`, `val_precision`, `val_recall`, `val_f1_score`, `val_roc_auc`,
    `val_pr_auc`, `val_positive_prevalence`, and confusion matrix counts (`val_true_negative`,
    `val_false_positive`, `val_false_negative`, `val_true_positive`).
  - **Test:** `test_accuracy`, `test_precision`, `test_recall`, `test_f1_score`, `test_roc_auc`,
    `test_pr_auc`, `test_positive_prevalence`, confusion matrix counts, and deltas vs. dummy baseline
    (`test_accuracy_vs_dummy`, `test_roc_auc_vs_dummy`, `test_pr_auc_vs_dummy`, `test_f1_vs_dummy`).
- **Model Artifact:** Serialized scikit-learn pipeline logged via `mlflow.sklearn.log_model` under `model`,
  including inferred input schema signature and input example.
- **Disk Artifacts:** All local artifacts mirrored under `artifacts/` (`model.joblib`, `dummy_model.joblib`,
  `evaluation_metrics.json`, `split_metadata.json`, and `evaluation_report.md`).

### Inspecting Runs Locally in the MLflow UI

Launch the MLflow user interface to compare runs, visualize ROC/PR metrics, inspect parameters, and download artifacts:

```powershell
uv run mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Open <http://localhost:5000> in your browser to inspect the tracked experiments.

## Real-Time Ingestion Pipeline with Kafka and Spark Structured Streaming (Phase 5)

Phase 5 introduces a reliable, locally runnable real-time streaming ingestion pipeline built
with Apache Kafka and PySpark Structured Streaming. The pipeline bridges pre-departure live flight
events with downstream analytics while upholding strict leakage prevention, schema validation,
and stateful stream deduplication.

### Streaming Architecture

```text
┌─────────────────────────────────┐
│ Synthetic / Live Event Producer │ (CLI: flight-delay-produce)
└────────────────┬────────────────┘
                 │ JSON Events (schema_version: 1.0.0)
                 ▼
┌─────────────────────────────────┐
│     Kafka: flight-events        │ (Port 9092 host / 29092 container)
└───────┬─────────────────┬───────┘
        │                 │ (Independent consumer streams)
        ▼                 ▼
┌──────────────────┐    ┌──────────────────────────────────────────────┐
│ Valid Stream Q1  │    │ Rejected Stream Q2 (Parquet) & Q3 (Kafka DLQ)│
│ Spark readStream │    │ Spark readStream                             │
│ Checkpoint:      │    │ Checkpoints: checkpoints/rejected            │
│ checkpoints/valid│    │              checkpoints/kafka_dlq           │
└───────┬──────────┘    └───────────────────────┬──────────────────────┘
        │ Valid Parquet                         │ Rejected Parquet & DLQ
        ▼                                       ▼
┌──────────────────┐            ┌──────────────────────────────────────┐
│  data/streaming/ │            │ Local Sink: data/streaming/rejected/ │
│  valid/          │            │ Kafka DLQ:  flight-events-dlq topic  │
└──────────────────┘            └──────────────────────────────────────┘
```

### Event Contract (`1.0.0`)

The streaming contract standardizes live pre-departure flight events, matching the historical BTS
15-feature standard:

```json
{
  "event_id": "8f38d4e1bf248d689b09930fec6561f7",
  "schema_version": "1.0.0",
  "event_timestamp": "2023-06-25T14:30:00Z",
  "ingestion_timestamp": "2023-06-25T14:30:05Z",
  "source": {
    "system": "synthetic-producer",
    "environment": "development",
    "is_synthetic": "true"
  },
  "carrier": "AA",
  "flight_number": "AA101",
  "origin": "DFW",
  "dest": "ORD",
  "fl_date": "2023-06-25",
  "crs_dep_time": "1430",
  "crs_dep_hour": 14,
  "crs_dep_minute": 30,
  "dep_time_blk": "1400-1459",
  "crs_arr_hour": 16,
  "crs_arr_minute": 55,
  "arr_time_blk": "1600-1659",
  "month": 6,
  "day_of_month": 25,
  "day_of_week": 7,
  "is_weekend": 1,
  "crs_elapsed_time": 145.0,
  "distance": 802.0
}
```

- **Envelope Metadata:**
  - `schema_version`: String identifier (fixed to `"1.0.0"`).
  - `event_id`: Deterministic 32-character SHA-256 hex digest calculated over normalized business keys:
    `sha256(f"{carrier}:{flight_number}:{origin}:{dest}:{fl_date}:{crs_dep_time}".encode("utf-8")).hexdigest()[:32]`.
    Cannot be null, empty, or whitespace-only.
  - `event_timestamp`: UTC ISO-8601 timestamp string (`YYYY-MM-DDTHH:MM:SSZ`).
  - `ingestion_timestamp`: UTC ISO-8601 timestamp recorded at publication time.
  - `source`: Map containing provenance metadata.
- **Business Payload (15 prediction features):**
  - Carrier & routing: `carrier` (2-3 char IATA), `flight_number`, `origin` (3-char IATA), `dest` (3-char IATA, $\ne \text{origin}$), `crs_elapsed_time` ($> 0.0$), `distance` ($> 0.0$).
  - Scheduled timing: `crs_dep_time` (`HHMM`), `crs_dep_hour` ($0\text{--}23$), `crs_dep_minute` ($0\text{--}59$), `dep_time_blk` (`HHMM-HHMM`), `crs_arr_hour` ($0\text{--}23$), `crs_arr_minute` ($0\text{--}59$), `arr_time_blk` (`HHMM-HHMM`).
  - Calendar: `fl_date` (`YYYY-MM-DD`), `month` ($1\text{--}12$), `day_of_month` ($1\text{--}31$), `day_of_week` ($1\text{--}7$, ISO standard), `is_weekend` ($0$ or $1$).

### Timestamp Semantics

To prevent timezone distortion and guarantee reliable watermarking:
1. **Local Airport Schedule Times:** `crs_dep_time`, `crs_dep_hour`, `crs_dep_minute`, `crs_arr_hour`, and
   `crs_arr_minute` represent **local airport scheduled times** exactly as published in airline timetables,
   matching historical BTS definitions.
2. **UTC Event Timestamps:** `event_timestamp` (and `ingestion_timestamp`) strictly follows a **UTC convention**
   (`...Z` or `+00:00`). Watermarking and streaming state expiration in Spark Structured Streaming operate
   exclusively on this parsed UTC timeline (`event_time`).

### Strict Leakage Prevention

Pre-departure prediction models must never receive post-departure actuals or target labels. The
following 14 columns and patterns are strictly prohibited by `PROHIBITED_LEAKAGE_COLUMNS`:
- `dep_delay`, `arr_delay`, `is_delayed_15`, `actual_elapsed_time`
- `dep_time`, `arr_time`, `wheels_off`, `wheels_on`, `air_time`
- `taxi_in`, `taxi_out`
- `cancelled`, `cancellation_code`, `diverted`

Streaming validation inspects raw JSON keys recursively (including nested dictionaries). If any prohibited
key is found, the event is immediately flagged as `LEAKAGE_DETECTED` and routed to the Dead-Letter Queue.

### Synthetic Kafka Producer

The producer (`flight-delay-produce`) generates realistic flight events modeled after BTS traffic patterns
and supports controlled error injection to test validation pipelines:

```powershell
# Run synthetic producer via Makefile (100 events at 10 events/sec)
make kafka-produce

# Or run via CLI with full options
uv run flight-delay-produce `
  --bootstrap-servers localhost:9092 `
  --topic flight-events `
  --rate 20.0 `
  --count 500 `
  --corrupt-prob 0.05

# Dry run mode (generates and validates events locally without connecting to Kafka)
uv run flight-delay-produce --dry-run --count 50 --corrupt-prob 0.1
```

Supported error injections:
- Malformed unclosed JSON strings (`MALFORMED_JSON`).
- Missing required fields (e.g. missing `carrier`).
- Out-of-bounds numerical fields (`crs_dep_hour: 99`).
- Post-departure leakage fields (`dep_delay: 45.0` or `dep_time: 1430`).
- Negative distance or duration.

### PySpark Structured Streaming Consumer

The streaming consumer (`flight-delay-stream`) processes events from Kafka, performs schema validation,
deduplicates within a watermark, writes valid records to Parquet, and routes invalid records to DLQ sinks:

```powershell
# Run streaming consumer via Makefile
make stream

# Or run via CLI with custom configuration and optional Kafka DLQ publishing
uv run flight-delay-stream `
  --bootstrap-servers localhost:9092 `
  --topic flight-events `
  --valid-output data/streaming/valid `
  --rejected-output data/streaming/rejected `
  --checkpoint-dir data/streaming/checkpoints `
  --watermark "10 minutes" `
  --trigger "5 seconds" `
  --publish-dlq `
  --dlq-topic flight-events-dlq
```

#### Dual Dead-Letter Queue (DLQ) Sinks
Invalid records (schema parse errors, missing keys, empty event IDs, boundary errors, or leakage fields)
are routed to:
1. **Local Parquet Sink (`data/streaming/rejected`):** Contains `event_id`, `raw_payload` (original unparsed string),
   `error_code`, `rejection_reason`, and `rejection_timestamp`.
2. **Kafka DLQ Topic (`flight-events-dlq`):** Enabled when `--publish-dlq` is supplied. Publishes dead-letter
   JSON records for real-time alerting.

### Delivery Semantics & Sink Failure Limitations

| Dimension | Mechanism | Guarantee / Behavior |
|---|---|---|
| **Kafka Ingestion** | Spark Kafka Source with checkpoint offsets | **At-least-once per query**. Committed offsets in write-ahead logs ensure no messages are lost across restarts. |
| **In-Window Deduplication** | 10-minute watermark on `event_time` + `dropDuplicates(["event_id", "event_time"])` | **Within-watermark deduplication**. Replayed records with identical `event_id` and `event_time` within 10 minutes are dropped. Different `event_time` values produce separate records. |
| **Late Data** | Spark Watermark | Records with `event_time < max(event_time) - 10 minutes` are dropped as late data. |
| **Parquet Sinks** | Append mode with atomic `_spark_metadata` commit logs | **Append-only**. Re-running the pipeline from scratch on existing output folders without clearing creates duplicate part-files (`make stream-clean` resets state). |

#### Important Sink Failure & Coordination Limitations
- **Independent Sinks:** Valid Parquet, rejected Parquet, and Kafka DLQ operate as **separate streaming queries**
  with distinct checkpoint subdirectories (`checkpoints/valid`, `checkpoints/rejected`, `checkpoints/kafka_dlq`).
- **No Atomic Cross-Sink Transactions:** Spark Structured Streaming does not provide two-phase commit across separate
  queries. If the Kafka DLQ broker fails or disk space is exhausted on one sink, that specific query will fail while
  the other query may continue advancing Kafka offsets.
- **Offset Desynchronization on Recovery:** After a partial sink failure, restarting the consumer resumes each query
  from its own last-committed checkpoint offset, which may lead to temporary offset skew across sinks.
- **End-to-End Exactly-Once Limitation:** The system does **not** provide end-to-end exactly-once delivery across
  all sinks. In Phase 6, transactional upserts to an online feature store (Cassandra) using primary keys will provide
  idempotent end-to-end writes.

### Makefile Commands for Streaming

```powershell
make kafka-up       # Start Kafka broker via Docker Compose
make kafka-topics   # Create flight-events and flight-events-dlq topics
make kafka-produce  # Produce 100 synthetic events to Kafka
make stream         # Start PySpark Structured Streaming consumer
make stream-clean   # Clear local streaming outputs and checkpoints
make kafka-down     # Stop Kafka container
```

## Next milestone

Phase 6: Online Feature Store (Cassandra) and Real-Time Prediction Serving (FastAPI integration with MLflow-trained models).
