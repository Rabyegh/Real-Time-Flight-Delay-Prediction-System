"""External flight data ingestion components."""

from flight_delay.ingestion.contract import (
    ContractValidationError,
    DeadLetterRecord,
    FlightEvent,
    contains_prohibited_leakage,
    generate_event_id,
    get_spark_dead_letter_schema,
    get_spark_flight_event_schema,
    validate_flight_event_dict,
)
from flight_delay.ingestion.kafka_producer import (
    KafkaProducerError,
    ProducerConfig,
    generate_synthetic_flight_event,
    run_producer,
)

__all__ = [
    "ContractValidationError",
    "DeadLetterRecord",
    "FlightEvent",
    "KafkaProducerError",
    "ProducerConfig",
    "contains_prohibited_leakage",
    "generate_event_id",
    "generate_synthetic_flight_event",
    "get_spark_dead_letter_schema",
    "get_spark_flight_event_schema",
    "run_producer",
    "validate_flight_event_dict",
]
