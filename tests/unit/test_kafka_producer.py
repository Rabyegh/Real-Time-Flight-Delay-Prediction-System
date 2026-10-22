"""Unit tests for synthetic Kafka flight event producer."""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from flight_delay.ingestion.kafka_producer import (
    KafkaProducerError,
    ProducerConfig,
    _build_argument_parser,
    generate_synthetic_flight_event,
    inject_corruption,
    main,
    run_producer,
)


def test_generate_synthetic_flight_event_custom_parameters() -> None:
    custom_date = datetime.date(2023, 6, 20)
    event = generate_synthetic_flight_event(
        seed=123,
        flight_date=custom_date,
        carrier="DL",
        flight_num=555,
        origin="ATL",
        dest="BOS",
        dep_hour=14,
        dep_min=30,
    )

    assert event.carrier == "DL"
    assert event.flight_number == "DL555"
    assert event.origin == "ATL"
    assert event.dest == "BOS"
    assert event.fl_date == "2023-06-20"
    assert event.crs_dep_hour == 14
    assert event.crs_dep_minute == 30
    assert event.dep_time_blk == "1400-1459"
    assert event.month == 6
    assert event.day_of_month == 20
    assert event.day_of_week == 2
    assert event.is_weekend == 0
    assert event.source["is_synthetic"] == "true"


def test_inject_corruption_produces_expected_error_types() -> None:
    base_event = generate_synthetic_flight_event(seed=1).to_dict()

    # Type 0: Malformed unparseable JSON
    payload0, reason0 = inject_corruption(base_event, 0)
    assert isinstance(payload0, str)
    assert reason0 == "MALFORMED_JSON"

    # Type 1: Missing field carrier
    payload1, reason1 = inject_corruption(base_event, 1)
    assert isinstance(payload1, dict)
    assert "carrier" not in payload1
    assert reason1 == "MISSING_FIELD_CARRIER"

    # Type 2: Out of bounds crs_dep_hour
    payload2, reason2 = inject_corruption(base_event, 2)
    assert isinstance(payload2, dict)
    assert payload2["crs_dep_hour"] == 99
    assert reason2 == "OUT_OF_BOUNDS_DEP_HOUR"

    # Type 3: Leakage field
    payload3, reason3 = inject_corruption(base_event, 3)
    assert isinstance(payload3, dict)
    assert "dep_delay" in payload3
    assert reason3 == "LEAKAGE_FIELD_DEP_DELAY"

    # Type 4: Negative distance
    payload4, reason4 = inject_corruption(base_event, 4)
    assert isinstance(payload4, dict)
    assert payload4["distance"] == -500.0
    assert reason4 == "NEGATIVE_DISTANCE"


def test_run_producer_dry_run_completes_with_accurate_counts() -> None:
    config = ProducerConfig(
        total_count=20,
        rate_per_sec=100.0,
        corrupt_prob=0.20,
        dry_run=True,
        random_seed=42,
    )
    stats = run_producer(config)

    assert stats["total_published"] == 20
    assert stats["valid_published"] > 0
    assert stats["corrupt_published"] > 0
    assert (
        stats["valid_published"] + stats["corrupt_published"] + stats["duplicate_published"] == 20
    )


def test_run_producer_with_mocked_kafka() -> None:
    config = ProducerConfig(
        bootstrap_servers="localhost:9092",
        topic="test-topic",
        total_count=5,
        rate_per_sec=100.0,
        corrupt_prob=0.0,
        dry_run=False,
        random_seed=42,
    )

    mock_producer_instance = MagicMock()
    with patch("kafka.KafkaProducer", return_value=mock_producer_instance):
        stats = run_producer(config)

    assert stats["valid_published"] == 5
    assert mock_producer_instance.send.call_count == 5
    assert mock_producer_instance.flush.called
    assert mock_producer_instance.close.called


def test_run_producer_fails_on_unreachable_broker() -> None:
    config = ProducerConfig(
        bootstrap_servers="nonexistent-broker:9092",
        topic="test-topic",
        total_count=5,
        dry_run=False,
    )

    with patch("kafka.KafkaProducer", side_effect=Exception("Connection refused")):
        with pytest.raises(KafkaProducerError, match="Failed to connect to Kafka broker"):
            run_producer(config)


def test_producer_argument_parser() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args(
        [
            "--bootstrap-servers",
            "broker:9092",
            "--topic",
            "custom-topic",
            "--rate",
            "25.0",
            "--count",
            "50",
            "--corrupt-prob",
            "0.1",
            "--dry-run",
            "--seed",
            "99",
        ]
    )

    assert args.bootstrap_servers == "broker:9092"
    assert args.topic == "custom-topic"
    assert args.rate == 25.0
    assert args.count == 50
    assert args.corrupt_prob == 0.1
    assert args.dry_run is True
    assert args.seed == 99


def test_producer_main_cli_dry_run() -> None:
    exit_code = main(["--dry-run", "--count", "5", "--rate", "100.0"])
    assert exit_code == 0
