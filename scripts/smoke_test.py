"""Check host-published local infrastructure endpoints."""

from __future__ import annotations

import socket
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from requests.exceptions import RequestException

from flight_delay.config.settings import get_settings


def check_http(url: str) -> None:
    with urlopen(url, timeout=3) as response:
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}")


def check_tcp(host: str, port: int) -> None:
    with socket.create_connection((host, port), timeout=3):
        pass


def check_mlflow_artifacts(tracking_uri: str) -> None:
    with urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=3) as response:
        if response.status >= 400:
            raise RuntimeError(f"MLflow HTTP {response.status}")

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("infrastructure-smoke")
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment("infrastructure-smoke")
    )
    run_id = client.create_run(experiment_id).info.run_id
    try:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact-check.txt"
            artifact.write_text("MinIO artifact write check\n", encoding="utf-8")
            client.log_artifact(run_id, str(artifact))
        artifacts = client.list_artifacts(run_id)
        if not any(item.path == "artifact-check.txt" for item in artifacts):
            raise RuntimeError("Uploaded artifact was not listed by MLflow")
    finally:
        client.set_terminated(run_id)


def main() -> int:
    settings = get_settings()
    checks: list[tuple[str, Callable[[], None]]] = [
        ("API", lambda: check_http(f"http://localhost:{settings.api_port}/health")),
        (
            "Kafka",
            lambda: check_tcp("localhost", settings.kafka_external_port),
        ),
        (
            "Spark master",
            lambda: check_http(
                f"http://localhost:{settings.spark_master_ui_port}",
            ),
        ),
        (
            "MinIO",
            lambda: check_http(
                f"http://localhost:{settings.minio_api_port}/minio/health/live",
            ),
        ),
        (
            "Cassandra",
            lambda: check_tcp("localhost", settings.cassandra_port_published),
        ),
        (
            "MLflow / MinIO artifact write",
            lambda: check_mlflow_artifacts(settings.mlflow_tracking_uri),
        ),
        (
            "Prometheus",
            lambda: check_http(
                "Prometheus",
                f"http://localhost:{settings.prometheus_port}/-/healthy",
            ),
        ),
        (
            "Grafana",
            lambda: check_http(
                "Grafana",
                f"http://localhost:{settings.grafana_port}/api/health",
            ),
        ),
    ]
    failures = 0
    for name, check in checks:
        try:
            check()
        except (
            MlflowException,
            OSError,
            RequestException,
            RuntimeError,
            URLError,
            ValueError,
        ) as error:
            failures += 1
            print(f"FAIL {name}: {error}")
        else:
            print(f"PASS {name}")
    print(f"{len(checks) - failures}/{len(checks)} services healthy")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
