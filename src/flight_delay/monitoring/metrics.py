"""Prometheus metrics for flight delay prediction platform."""

from __future__ import annotations

import logging

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

# Bounded histogram buckets
HTTP_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
INFERENCE_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
STORAGE_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
STREAMING_DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class FlightDelayMetrics:
    """Isolated Prometheus metrics container and recorder.

    Accepts a custom CollectorRegistry to avoid shared global registry state in tests.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry: CollectorRegistry = (
            registry if registry is not None else CollectorRegistry(auto_describe=True)
        )

        # Ingestion metrics
        self.events_received = Counter(
            "flight_delay_events_received_total",
            "Total flight events received by ingestion",
            ["source"],
            registry=self.registry,
        )
        self.events_valid = Counter(
            "flight_delay_events_valid_total",
            "Total valid flight events accepted by pipeline",
            registry=self.registry,
        )
        self.events_rejected = Counter(
            "flight_delay_events_rejected_total",
            "Total flight events rejected during schema validation or leakage detection",
            ["reason"],
            registry=self.registry,
        )
        self.kafka_errors = Counter(
            "flight_delay_kafka_errors_total",
            "Total Kafka processing or publishing errors",
            ["error_type"],
            registry=self.registry,
        )
        self.dead_letter_events = Counter(
            "flight_delay_dead_letter_events_total",
            "Total events routed to dead-letter queue",
            ["reason"],
            registry=self.registry,
        )

        # Streaming metrics
        self.streaming_microbatch_duration = Histogram(
            "flight_delay_streaming_microbatch_duration_seconds",
            "Spark Structured Streaming micro-batch execution duration in seconds",
            buckets=STREAMING_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.streaming_input_rows_per_second = Gauge(
            "flight_delay_streaming_input_rows_per_second",
            "Recent micro-batch input rows per second",
            registry=self.registry,
        )
        self.streaming_processed_rows_per_second = Gauge(
            "flight_delay_streaming_processed_rows_per_second",
            "Recent micro-batch processed rows per second",
            registry=self.registry,
        )
        self.streaming_query_status = Gauge(
            "flight_delay_streaming_query_status",
            "Streaming query execution status (1=running, 0=stopped)",
            ["query_name"],
            registry=self.registry,
        )
        self.streaming_query_failures = Counter(
            "flight_delay_streaming_query_failures_total",
            "Total streaming query unexpected termination failures",
            ["query_name"],
            registry=self.registry,
        )

        # Model inference metrics
        self.predictions = Counter(
            "flight_delay_predictions_total",
            "Total flight delay predictions evaluated",
            ["model_version", "status"],
            registry=self.registry,
        )
        self.prediction_latency = Histogram(
            "flight_delay_prediction_latency_seconds",
            "Model inference execution latency in seconds",
            buckets=INFERENCE_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.model_version_info = Gauge(
            "flight_delay_model_version_info",
            "Metadata for active model version and threshold",
            ["version", "threshold"],
            registry=self.registry,
        )
        self.prediction_errors = Counter(
            "flight_delay_prediction_errors_total",
            "Total errors encountered during inference evaluation",
            ["error_type"],
            registry=self.registry,
        )

        # Cassandra storage metrics
        self.cassandra_writes = Counter(
            "flight_delay_cassandra_writes_total",
            "Total Cassandra persistence writes attempted",
            ["table", "status"],
            registry=self.registry,
        )
        self.cassandra_write_latency = Histogram(
            "flight_delay_cassandra_write_latency_seconds",
            "Cassandra persistence write latency in seconds",
            buckets=STORAGE_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.cassandra_connectivity_status = Gauge(
            "flight_delay_cassandra_connectivity_status",
            "Cassandra cluster connectivity status (1=connected, 0=disconnected)",
            registry=self.registry,
        )
        # Default connectivity to 0 (disconnected)
        self.cassandra_connectivity_status.set(0)

        # API serving metrics
        self.api_requests = Counter(
            "flight_delay_api_requests_total",
            "Total HTTP requests received by API",
            ["endpoint", "method", "status_code"],
            registry=self.registry,
        )
        self.api_request_latency = Histogram(
            "flight_delay_api_request_latency_seconds",
            "API request duration in seconds",
            ["endpoint", "method"],
            buckets=HTTP_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.api_errors = Counter(
            "flight_delay_api_errors_total",
            "Total errors or 5xx exceptions in API handling",
            ["endpoint", "error_type"],
            registry=self.registry,
        )

    # -------------------------------------------------------------------------
    # Safe recording helper methods (never crash calling application)
    # -------------------------------------------------------------------------

    def record_api_request(
        self, endpoint: str, method: str, status_code: int, duration_seconds: float
    ) -> None:
        """Record an API request count and latency."""
        try:
            self.api_requests.labels(
                endpoint=endpoint, method=method, status_code=str(status_code)
            ).inc()
            if duration_seconds >= 0:
                self.api_request_latency.labels(endpoint=endpoint, method=method).observe(
                    duration_seconds
                )
        except Exception as exc:
            logger.debug("Failed to record api_request metric: %s", exc)

    def record_api_error(self, endpoint: str, error_type: str) -> None:
        """Record an API error occurrence."""
        try:
            self.api_errors.labels(endpoint=endpoint, error_type=error_type).inc()
        except Exception as exc:
            logger.debug("Failed to record api_error metric: %s", exc)

    def record_prediction(
        self,
        model_version: str,
        is_delayed: bool,
        duration_seconds: float,
        threshold: float | None = None,
    ) -> None:
        """Record model inference count, classification label, and latency."""
        try:
            status_label = "delayed" if is_delayed else "not_delayed"
            self.predictions.labels(model_version=model_version, status=status_label).inc()
            if duration_seconds >= 0:
                self.prediction_latency.observe(duration_seconds)
            if threshold is not None:
                self.model_version_info.labels(version=model_version, threshold=str(threshold)).set(
                    1
                )
        except Exception as exc:
            logger.debug("Failed to record prediction metric: %s", exc)

    def record_prediction_error(self, error_type: str) -> None:
        """Record a model inference error."""
        try:
            self.prediction_errors.labels(error_type=error_type).inc()
        except Exception as exc:
            logger.debug("Failed to record prediction_error metric: %s", exc)

    def record_cassandra_write(
        self, table: str, success: bool, duration_seconds: float | None = None
    ) -> None:
        """Record Cassandra write result and optional latency."""
        try:
            status_label = "success" if success else "failure"
            self.cassandra_writes.labels(table=table, status=status_label).inc()
            if duration_seconds is not None and duration_seconds >= 0:
                self.cassandra_write_latency.observe(duration_seconds)
        except Exception as exc:
            logger.debug("Failed to record cassandra_write metric: %s", exc)

    def set_cassandra_status(self, is_connected: bool) -> None:
        """Update Cassandra connection health gauge (1=connected, 0=disconnected)."""
        try:
            self.cassandra_connectivity_status.set(1 if is_connected else 0)
        except Exception as exc:
            logger.debug("Failed to set cassandra_status metric: %s", exc)

    def record_ingestion_event(
        self, source: str = "kafka", is_valid: bool = True, reason: str = ""
    ) -> None:
        """Record ingested event counts."""
        try:
            self.events_received.labels(source=source).inc()
            if is_valid:
                self.events_valid.inc()
            else:
                self.events_rejected.labels(reason=reason or "unknown").inc()
        except Exception as exc:
            logger.debug("Failed to record ingestion_event metric: %s", exc)

    def record_kafka_error(self, error_type: str) -> None:
        """Record a Kafka publish or consume error."""
        try:
            self.kafka_errors.labels(error_type=error_type).inc()
        except Exception as exc:
            logger.debug("Failed to record kafka_error metric: %s", exc)

    def record_dead_letter_event(self, reason: str) -> None:
        """Record an event pushed to DLQ."""
        try:
            self.dead_letter_events.labels(reason=reason or "unknown").inc()
        except Exception as exc:
            logger.debug("Failed to record dead_letter_event metric: %s", exc)

    def record_streaming_microbatch(
        self,
        duration_seconds: float,
        input_rows_per_second: float | None = None,
        processed_rows_per_second: float | None = None,
    ) -> None:
        """Record Spark Structured Streaming micro-batch metrics."""
        try:
            if duration_seconds >= 0:
                self.streaming_microbatch_duration.observe(duration_seconds)
            if input_rows_per_second is not None:
                self.streaming_input_rows_per_second.set(input_rows_per_second)
            if processed_rows_per_second is not None:
                self.streaming_processed_rows_per_second.set(processed_rows_per_second)
        except Exception as exc:
            logger.debug("Failed to record streaming_microbatch metric: %s", exc)

    def set_streaming_query_status(self, query_name: str, is_running: bool) -> None:
        """Set streaming query active status."""
        try:
            self.streaming_query_status.labels(query_name=query_name).set(1 if is_running else 0)
        except Exception as exc:
            logger.debug("Failed to set streaming_query_status metric: %s", exc)

    def record_streaming_query_failure(self, query_name: str) -> None:
        """Record a streaming query termination failure."""
        try:
            self.streaming_query_failures.labels(query_name=query_name).inc()
        except Exception as exc:
            logger.debug("Failed to record streaming_query_failure metric: %s", exc)

    def generate_metrics_text(self) -> tuple[bytes, str]:
        """Generate Prometheus-formatted byte output and content type."""
        try:
            return generate_latest(self.registry), CONTENT_TYPE_LATEST
        except Exception as exc:
            logger.error("Failed to generate Prometheus metrics output: %s", exc)
            return b"# Error generating metrics\n", "text/plain; charset=utf-8"
