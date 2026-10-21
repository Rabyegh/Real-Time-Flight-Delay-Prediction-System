# Real-Time Flight Delay Prediction System

A local development foundation for a flight-delay platform using US DOT Bureau of
Transportation Statistics (BTS) history and AviationStack live flight data. The
repository is intentionally a scaffold: ingestion, ETL, streaming transformations,
feature engineering, model training, and prediction-serving integrations are not
implemented yet.

## Architecture

```text
BTS historical data ─┐
                     ├─ future ingestion → Kafka → Spark Structured Streaming
AviationStack ───────┘                              ├─ MinIO (Parquet datasets)
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
   uv run pytest
   ```

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
  processing/           Future historical ETL, streaming, and features
  storage/              Future Cassandra and dataset persistence
  training/             Future classifier/regressor training and evaluation
  api/                  FastAPI application
tests/unit/             Configuration and API tests
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

Confirm data licensing and BTS download details, then agree on the raw flight
record schema and AviationStack polling/refresh policy before implementing
idempotent ingestion and Kafka event contracts. The schema and model semantics
are intentionally not invented in this scaffold.
