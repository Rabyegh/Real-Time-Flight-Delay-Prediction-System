"""Kafka producer for publishing synthetic pre-departure flight events.

Timestamp Semantics:
- `crs_dep_time`, `crs_dep_hour`, `crs_dep_minute`, `crs_arr_hour`, and `crs_arr_minute`
  represent local airport scheduled departure/arrival times, conforming to the historical
  BTS standard and matching the Phase 2/3 feature definition.
- `event_timestamp` and `ingestion_timestamp` represent streaming event occurrence and ingestion
  times in UTC (ISO 8601 with trailing 'Z'), which are used for Spark Structured Streaming
  watermarking and stateful deduplication.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from flight_delay.config.settings import get_settings
from flight_delay.ingestion.contract import (
    SCHEMA_VERSION,
    FlightEvent,
    generate_event_id,
)

LOGGER = logging.getLogger(__name__)

CARRIERS: tuple[str, ...] = ("AA", "DL", "UA", "WN", "B6", "AS", "NK")

AIRPORTS: tuple[str, ...] = (
    "ATL",
    "BOS",
    "CLT",
    "DEN",
    "DFW",
    "DTW",
    "EWR",
    "IAH",
    "JFK",
    "LAS",
    "LAX",
    "LGA",
    "MCO",
    "MIA",
    "MSP",
    "ORD",
    "PHL",
    "PHX",
    "SEA",
    "SFO",
)


class KafkaProducerError(RuntimeError):
    """Raised when Kafka producer fails to initialize or publish events."""


@dataclass
class ProducerConfig:
    """Configuration parameters for the synthetic flight event producer."""

    bootstrap_servers: str = "localhost:9092"
    topic: str = "flight-events"
    rate_per_sec: float = 10.0
    total_count: int | None = 100
    corrupt_prob: float = 0.05
    dry_run: bool = False
    random_seed: int | None = 42


def generate_synthetic_flight_event(
    seed: int | None = None,
    flight_date: datetime.date | None = None,
    carrier: str | None = None,
    flight_num: int | None = None,
    origin: str | None = None,
    dest: str | None = None,
    dep_hour: int | None = None,
    dep_min: int | None = None,
) -> FlightEvent:
    """Generate a realistic, leakage-free synthetic pre-departure flight event."""
    rng = random.Random(seed) if seed is not None else random

    fl_date = flight_date or datetime.date(2023, 6, 25)
    c_carrier = carrier or rng.choice(CARRIERS)
    c_flight_num = flight_num or rng.randint(100, 2999)
    flight_number_str = f"{c_carrier}{c_flight_num}"

    c_origin = origin or rng.choice(AIRPORTS)
    remaining_airports = [a for a in AIRPORTS if a != c_origin]
    c_dest = dest or rng.choice(remaining_airports)

    c_dep_hour = dep_hour if dep_hour is not None else rng.randint(5, 22)
    c_dep_min = dep_min if dep_min is not None else rng.choice((0, 15, 30, 45, rng.randint(0, 59)))
    crs_dep_time_str = f"{c_dep_hour:02d}{c_dep_min:02d}"
    dep_blk = f"{c_dep_hour:02d}00-{c_dep_hour:02d}59"

    # Realistic elapsed time and distance
    distance_val = round(rng.uniform(200.0, 2600.0), 1)
    elapsed_minutes = round((distance_val / 7.5) + rng.uniform(35.0, 65.0), 1)

    # Compute scheduled arrival in local clock terms
    total_dep_minutes = c_dep_hour * 60 + c_dep_min
    total_arr_minutes = int(total_dep_minutes + elapsed_minutes) % (24 * 60)
    c_arr_hour = total_arr_minutes // 60
    c_arr_min = total_arr_minutes % 60
    arr_blk = f"{c_arr_hour:02d}00-{c_arr_hour:02d}59"

    # Calendar features
    month = fl_date.month
    day_of_month = fl_date.day
    day_of_week = fl_date.isoweekday()  # 1=Monday, 7=Sunday
    is_weekend = 1 if day_of_week in (6, 7) else 0

    # Deterministic event identity
    event_id = generate_event_id(
        carrier=c_carrier,
        flight_number=flight_number_str,
        origin=c_origin,
        dest=c_dest,
        fl_date=fl_date.isoformat(),
        crs_dep_time=crs_dep_time_str,
    )

    # Event timestamps in UTC
    now_utc = datetime.datetime.now(datetime.UTC)
    now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Planned pre-departure event timestamp
    event_ts = datetime.datetime(
        year=fl_date.year,
        month=fl_date.month,
        day=fl_date.day,
        hour=c_dep_hour,
        minute=c_dep_min,
        tzinfo=datetime.UTC,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    return FlightEvent(
        event_id=event_id,
        schema_version=SCHEMA_VERSION,
        event_timestamp=event_ts,
        ingestion_timestamp=now_iso,
        source={
            "system": "synthetic-producer",
            "environment": "development",
            "is_synthetic": "true",
        },
        carrier=c_carrier,
        flight_number=flight_number_str,
        origin=c_origin,
        dest=c_dest,
        fl_date=fl_date.isoformat(),
        crs_dep_time=crs_dep_time_str,
        crs_dep_hour=c_dep_hour,
        crs_dep_minute=c_dep_min,
        dep_time_blk=dep_blk,
        crs_arr_hour=c_arr_hour,
        crs_arr_minute=c_arr_min,
        arr_time_blk=arr_blk,
        month=month,
        day_of_month=day_of_month,
        day_of_week=day_of_week,
        is_weekend=is_weekend,
        crs_elapsed_time=elapsed_minutes,
        distance=distance_val,
    )


def inject_corruption(
    event_dict: dict[str, Any], corruption_type: int
) -> tuple[str | dict[str, Any], str]:
    """Inject a specific error pattern for dead-letter testing."""
    corrupted = dict(event_dict)
    if corruption_type == 0:
        # 1. Malformed unparseable JSON string
        return (
            '{"event_id": "malformed", "carrier": "AA", "incomplete_json": true',
            "MALFORMED_JSON",
        )
    elif corruption_type == 1:
        # 2. Missing required field
        corrupted.pop("carrier", None)
        return corrupted, "MISSING_FIELD_CARRIER"
    elif corruption_type == 2:
        # 3. Invalid out-of-bounds numeric field
        corrupted["crs_dep_hour"] = 99
        return corrupted, "OUT_OF_BOUNDS_DEP_HOUR"
    elif corruption_type == 3:
        # 4. Prohibited post-departure leakage field
        corrupted["dep_delay"] = 45.0
        return corrupted, "LEAKAGE_FIELD_DEP_DELAY"
    else:
        # 5. Negative distance
        corrupted["distance"] = -500.0
        return corrupted, "NEGATIVE_DISTANCE"


def run_producer(
    config: ProducerConfig,
    stop_event: Any | None = None,
) -> dict[str, int]:
    """Execute the synthetic Kafka producer publishing loop."""
    if config.random_seed is not None:
        random.seed(config.random_seed)

    producer = None
    if not config.dry_run:
        try:
            from kafka import KafkaProducer

            producer = KafkaProducer(
                bootstrap_servers=config.bootstrap_servers,
                value_serializer=lambda v: (
                    v.encode("utf-8") if isinstance(v, str) else json.dumps(v).encode("utf-8")
                ),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
            )
            LOGGER.info(
                "Connected to Kafka broker at %s, target topic: %s",
                config.bootstrap_servers,
                config.topic,
            )
        except Exception as exc:
            raise KafkaProducerError(
                f"Failed to connect to Kafka broker at '{config.bootstrap_servers}': {exc}"
            ) from exc

    published_valid = 0
    published_corrupt = 0
    published_duplicate = 0
    total_target = config.total_count

    sleep_interval = 1.0 / max(config.rate_per_sec, 0.1)
    recent_events: list[dict[str, Any]] = []

    LOGGER.info(
        "Starting synthetic producer (rate=%.1f/s, total=%s, corrupt_prob=%.2f, dry_run=%s)",
        config.rate_per_sec,
        str(total_target),
        config.corrupt_prob,
        config.dry_run,
    )

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                LOGGER.info("Stop event signaled; terminating producer.")
                break
            if (
                total_target is not None
                and (published_valid + published_corrupt + published_duplicate) >= total_target
            ):
                LOGGER.info("Reached target total event count: %d", total_target)
                break

            # Decide whether to inject corruption or duplicate
            should_corrupt = random.random() < config.corrupt_prob
            should_duplicate = len(recent_events) >= 5 and random.random() < 0.10

            if should_duplicate:
                # Replay an existing event with identical event_id
                event_to_replay = random.choice(recent_events)
                key = event_to_replay["event_id"]
                if producer:
                    producer.send(config.topic, key=key, value=event_to_replay)
                published_duplicate += 1
                LOGGER.debug("Published duplicate event_id=%s", key)
            elif should_corrupt:
                event = generate_synthetic_flight_event()
                corruption_idx = random.randint(0, 4)
                corrupted_payload, reason = inject_corruption(event.to_dict(), corruption_idx)
                key = (
                    corrupted_payload.get("event_id", "corrupt_key")
                    if isinstance(corrupted_payload, dict)
                    else "corrupt_key"
                )
                if producer:
                    producer.send(config.topic, key=key, value=corrupted_payload)
                published_corrupt += 1
                LOGGER.debug("Published corrupt event (%s)", reason)
            else:
                event = generate_synthetic_flight_event()
                event_dict = event.to_dict()
                key = event.event_id
                if producer:
                    producer.send(config.topic, key=key, value=event_dict)
                published_valid += 1
                recent_events.append(event_dict)
                if len(recent_events) > 20:
                    recent_events.pop(0)

            time.sleep(sleep_interval)

    except KeyboardInterrupt:
        LOGGER.info("Received KeyboardInterrupt; stopping producer gracefully.")
    finally:
        if producer:
            producer.flush()
            producer.close()
            LOGGER.info("Kafka producer flushed and closed.")

    stats = {
        "valid_published": published_valid,
        "corrupt_published": published_corrupt,
        "duplicate_published": published_duplicate,
        "total_published": published_valid + published_corrupt + published_duplicate,
    }
    LOGGER.info("Producer finished with stats: %s", stats)
    return stats


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for synthetic Kafka flight event producer."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Publish synthetic pre-departure flight events to Kafka for testing."
    )
    parser.add_argument(
        "--bootstrap-servers",
        "--broker",
        default=settings.kafka_bootstrap_servers,
        help=f"Kafka bootstrap servers (default: {settings.kafka_bootstrap_servers})",
    )
    parser.add_argument(
        "--topic",
        default=settings.kafka_flight_topic,
        help=f"Kafka topic for flight events (default: {settings.kafka_flight_topic})",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Event publication rate in events/second (default: 10.0)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Total events to produce, or -1 for continuous (default: 100)",
    )
    parser.add_argument(
        "--corrupt-prob",
        type=float,
        default=0.05,
        help="Probability of injecting corrupt/invalid records (default: 0.05)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and validate events without connecting to a live Kafka broker",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible event generation (default: 42)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for flight-delay-produce."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    total_count = None if args.count < 0 else args.count
    config = ProducerConfig(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        rate_per_sec=args.rate,
        total_count=total_count,
        corrupt_prob=args.corrupt_prob,
        dry_run=args.dry_run,
        random_seed=args.seed,
    )

    try:
        stats = run_producer(config)
        print(json.dumps(stats, indent=2))
        return 0
    except KafkaProducerError as err:
        LOGGER.error("Kafka producer execution failed: %s", err)
        return 2
    except Exception:
        LOGGER.exception("Unexpected error in synthetic producer")
        return 1


if __name__ == "__main__":
    sys.exit(main())
