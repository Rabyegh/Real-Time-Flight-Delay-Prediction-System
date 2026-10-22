"""Historical, feature engineering, and streaming data processing components."""

from flight_delay.processing.feature_engineering import (
    DELAY_THRESHOLD_MINUTES,
    FEATURE_ALLOWLIST,
    PROHIBITED_COLUMNS,
    TARGET_COLUMN,
    FeatureConfig,
    FeatureEngineeringError,
    TargetLeakageError,
    assert_no_leakage,
    build_feature_dataset,
    validate_dataset_paths,
)
from flight_delay.processing.streaming_consumer import (
    PathSafetyError,
    StreamingConfig,
    StreamingPipelineError,
    build_streaming_query,
    create_streaming_spark_session,
    parse_and_validate_stream,
    validate_streaming_paths,
)

__all__ = [
    "DELAY_THRESHOLD_MINUTES",
    "FEATURE_ALLOWLIST",
    "PROHIBITED_COLUMNS",
    "PathSafetyError",
    "StreamingConfig",
    "StreamingPipelineError",
    "TARGET_COLUMN",
    "TargetLeakageError",
    "assert_no_leakage",
    "build_feature_dataset",
    "build_streaming_query",
    "create_streaming_spark_session",
    "parse_and_validate_stream",
    "validate_dataset_paths",
    "validate_streaming_paths",
    "FeatureConfig",
    "FeatureEngineeringError",
]
