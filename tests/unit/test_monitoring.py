"""Unit tests for monitoring, observability, and Prometheus metrics."""

from __future__ import annotations

from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from flight_delay.api.main import create_app
from flight_delay.monitoring.metrics import FlightDelayMetrics


def test_metrics_registry_isolation() -> None:
    """Verify two FlightDelayMetrics instances do not share a global registry or collide."""
    reg1 = CollectorRegistry()
    reg2 = CollectorRegistry()

    m1 = FlightDelayMetrics(registry=reg1)
    m2 = FlightDelayMetrics(registry=reg2)

    m1.record_prediction(model_version="1.0.0", is_delayed=True, duration_seconds=0.01)
    m2.record_prediction(model_version="1.0.0", is_delayed=False, duration_seconds=0.02)

    text1, _ = m1.generate_metrics_text()
    text2, _ = m2.generate_metrics_text()

    assert b'status="delayed"' in text1
    assert b'status="not_delayed"' not in text1

    assert b'status="not_delayed"' in text2
    assert b'status="delayed"' not in text2


def test_ingestion_and_streaming_metrics() -> None:
    """Verify ingestion and streaming metrics recording."""
    metrics = FlightDelayMetrics(registry=CollectorRegistry())

    # Ingestion
    metrics.record_ingestion_event(source="kafka", is_valid=True)
    metrics.record_ingestion_event(source="kafka", is_valid=False, reason="schema_violation")
    metrics.record_kafka_error(error_type="broker_timeout")
    metrics.record_dead_letter_event(reason="unparseable_json")

    # Streaming
    metrics.record_streaming_microbatch(
        duration_seconds=0.45,
        input_rows_per_second=120.5,
        processed_rows_per_second=118.0,
    )
    metrics.set_streaming_query_status("predictions_sink", is_running=True)
    metrics.record_streaming_query_failure("predictions_sink")

    raw_text, _ = metrics.generate_metrics_text()
    text = raw_text.decode("utf-8")

    assert "flight_delay_events_received_total" in text
    assert "flight_delay_events_valid_total" in text
    assert 'flight_delay_events_rejected_total{reason="schema_violation"}' in text
    assert 'flight_delay_kafka_errors_total{error_type="broker_timeout"}' in text
    assert 'flight_delay_dead_letter_events_total{reason="unparseable_json"}' in text
    assert "flight_delay_streaming_microbatch_duration_seconds" in text
    assert "flight_delay_streaming_input_rows_per_second 120.5" in text
    assert "flight_delay_streaming_processed_rows_per_second 118.0" in text
    assert 'flight_delay_streaming_query_status{query_name="predictions_sink"} 1.0' in text
    assert 'flight_delay_streaming_query_failures_total{query_name="predictions_sink"}' in text


def test_cassandra_metrics() -> None:
    """Verify Cassandra persistence and connectivity metrics."""
    metrics = FlightDelayMetrics(registry=CollectorRegistry())

    metrics.record_cassandra_write(
        table="predictions_by_event", success=True, duration_seconds=0.005
    )
    metrics.record_cassandra_write(
        table="predictions_by_airport", success=False, duration_seconds=0.02
    )
    metrics.set_cassandra_status(True)

    raw_text, _ = metrics.generate_metrics_text()
    text = raw_text.decode("utf-8")

    assert (
        'flight_delay_cassandra_writes_total{status="success",table="predictions_by_event"}' in text
    )
    assert (
        'flight_delay_cassandra_writes_total{status="failure",table="predictions_by_airport"}'
        in text
    )
    assert "flight_delay_cassandra_connectivity_status 1.0" in text


def test_api_metrics_and_endpoint() -> None:
    """Verify API request recording and /metrics HTTP endpoint output."""
    metrics = FlightDelayMetrics(registry=CollectorRegistry())
    app = create_app(metrics=metrics)

    with TestClient(app) as client:
        # Generate some API traffic
        client.get("/health")
        client.get("/health")
        client.get("/api/v1/predictions/dummy_test_event")

        # Now scrape /metrics
        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "text/plain" in metrics_resp.headers["content-type"]
        text = metrics_resp.text

        assert "flight_delay_api_requests_total" in text
        assert 'endpoint="/health"' in text
        assert "flight_delay_api_request_latency_seconds" in text


def test_safe_metrics_recording_on_errors() -> None:
    """Verify metrics helpers do not raise exceptions even with unexpected inputs."""
    metrics = FlightDelayMetrics(registry=CollectorRegistry())
    # Negative latencies or extreme types should be handled gracefully without crashing
    metrics.record_prediction(model_version="1.0.0", is_delayed=True, duration_seconds=-1.0)
    metrics.record_api_request(
        endpoint="/test", method="GET", status_code=200, duration_seconds=-5.0
    )
    metrics.record_cassandra_write(table="tbl", success=True, duration_seconds=-0.5)
    metrics.record_streaming_microbatch(duration_seconds=-1.0)
