"""Unit tests for the model inference engine (ModelPredictor).

Validates feature contract enforcement, leakage rejection, threshold behavior,
model artifact loading, and prediction structure.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flight_delay.training.inference import (
    CATEGORICAL_COLUMNS,
    DEFAULT_MODEL_PATH,
    EXPECTED_FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
    ModelArtifactNotFoundError,
    ModelLoadError,
    ModelPredictor,
    PredictionResult,
    SchemaValidationError,
    TargetLeakageDetectedError,
)


@pytest.fixture
def valid_event_dict() -> dict[str, Any]:
    """Return a representative dictionary with all 15 required features and envelope fields."""
    return {
        "event_id": "test-event-001",
        "flight_number": "100",
        "carrier": "AA",
        "origin": "JFK",
        "dest": "LAX",
        "crs_dep_hour": 10,
        "crs_dep_minute": 30,
        "dep_time_blk": "1000-1059",
        "crs_arr_hour": 13,
        "crs_arr_minute": 45,
        "arr_time_blk": "1300-1359",
        "month": 6,
        "day_of_month": 15,
        "day_of_week": 4,
        "is_weekend": 0,
        "crs_elapsed_time": 375.0,
        "distance": 2475.0,
    }


@pytest.fixture
def synthetic_pipeline_path(tmp_path: Path) -> Path:
    """Create a minimal fitted scikit-learn Pipeline matching the baseline schema."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_COLUMNS),
            ),
            ("num", StandardScaler(), list(NUMERICAL_COLUMNS)),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42, solver="lbfgs")),
        ]
    )

    # Minimal synthetic training dataset (10 rows)
    data = []
    carriers = ["AA", "DL", "UA"]
    origins = ["JFK", "ORD", "ATL"]
    dests = ["LAX", "DFW", "DEN"]
    for i in range(10):
        data.append(
            {
                "carrier": carriers[i % len(carriers)],
                "origin": origins[i % len(origins)],
                "dest": dests[i % len(dests)],
                "crs_dep_hour": 8 + i,
                "crs_dep_minute": 15,
                "dep_time_blk": f"{8 + i:02d}00-{8 + i:02d}59",
                "crs_arr_hour": 11 + i,
                "crs_arr_minute": 30,
                "arr_time_blk": f"{11 + i:02d}00-{11 + i:02d}59",
                "month": 6,
                "day_of_month": 1 + i,
                "day_of_week": (i % 7) + 1,
                "is_weekend": 1 if (i % 7) in (5, 6) else 0,
                "crs_elapsed_time": 200.0 + i * 10,
                "distance": 1000.0 + i * 50,
            }
        )
    df_train = pd.DataFrame(data)[list(EXPECTED_FEATURE_COLUMNS)]
    y_train = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])

    pipeline.fit(df_train, y_train)

    artifact_file = tmp_path / "synthetic_model.joblib"
    joblib.dump(pipeline, artifact_file)
    return artifact_file


# -----------------------------------------------------------------------------
# 1. Real Artifact & Loading Tests
# -----------------------------------------------------------------------------


def test_load_real_model_artifact() -> None:
    """Verify loading the real trained baseline model artifact when present."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("Default model artifact models/baseline/model.joblib is not present.")

    predictor = ModelPredictor()
    assert predictor.feature_names == EXPECTED_FEATURE_COLUMNS
    assert len(predictor.feature_names) == 15
    assert predictor.threshold == 0.5
    assert predictor.model_version == "baseline-v1"


def test_missing_model_artifact(tmp_path: Path) -> None:
    """Ensure ModelArtifactNotFoundError is raised when model file does not exist."""
    missing_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(ModelArtifactNotFoundError) as exc_info:
        ModelPredictor(model_path=missing_path)
    assert "Model artifact not found" in str(exc_info.value)


def test_corrupt_model_artifact(tmp_path: Path) -> None:
    """Ensure ModelLoadError is raised when model file contains corrupt bytes."""
    corrupt_file = tmp_path / "corrupt_model.joblib"
    corrupt_file.write_bytes(b"INVALID_HEADER_GARBAGE_BYTES_123456789")
    with pytest.raises(ModelLoadError) as exc_info:
        ModelPredictor(model_path=corrupt_file)
    assert "Failed to deserialize model artifact" in str(exc_info.value)


def test_model_without_predict_proba(tmp_path: Path) -> None:
    """Ensure ModelLoadError is raised if model does not implement predict_proba."""
    dummy_obj = {"not_a_model": 123}
    path = tmp_path / "no_proba_model.joblib"
    joblib.dump(dummy_obj, path)

    with pytest.raises(ModelLoadError) as exc_info:
        ModelPredictor(model_path=path)
    assert "does not implement predict_proba" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 2. Prediction Behavior & Thresholds
# -----------------------------------------------------------------------------


def test_predict_representative_valid_event(
    synthetic_pipeline_path: Path, valid_event_dict: dict[str, Any]
) -> None:
    """Test inference on a valid event producing a valid finite probability."""
    # Use real model if available, else synthetic pipeline
    model_path = DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else synthetic_pipeline_path
    predictor = ModelPredictor(model_path=model_path)

    results = predictor.predict(valid_event_dict)
    assert len(results) == 1
    res = results[0]

    assert isinstance(res, PredictionResult)
    assert isinstance(res.delay_probability, float)
    assert 0.0 <= res.delay_probability <= 1.0
    assert res.predicted_delayed in (0, 1)
    assert res.event_id == "test-event-001"
    assert res.model_version == "baseline-v1"

    # Verify prediction_time is valid ISO 8601
    dt = datetime.datetime.fromisoformat(res.prediction_time.replace("Z", "+00:00"))
    assert dt is not None


def test_predict_single_convenience_method(
    synthetic_pipeline_path: Path, valid_event_dict: dict[str, Any]
) -> None:
    """Test predict_single with both dict and pd.Series."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)

    # 1. From dict
    res_dict = predictor.predict_single(valid_event_dict)
    assert isinstance(res_dict, PredictionResult)
    assert 0.0 <= res_dict.delay_probability <= 1.0

    # 2. From Series
    res_series = predictor.predict_single(pd.Series(valid_event_dict))
    assert isinstance(res_series, PredictionResult)
    assert res_series.delay_probability == res_dict.delay_probability


def test_predict_dataframe_enriched(
    synthetic_pipeline_path: Path, valid_event_dict: dict[str, Any]
) -> None:
    """Test predict_dataframe appending prediction columns to input DataFrame."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)
    df = pd.DataFrame([valid_event_dict, valid_event_dict])

    enriched_df = predictor.predict_dataframe(df)
    assert len(enriched_df) == 2
    assert "delay_probability" in enriched_df.columns
    assert "predicted_delayed" in enriched_df.columns
    assert "prediction_time" in enriched_df.columns
    assert "model_version" in enriched_df.columns

    # Verify original fields preserved
    assert "event_id" in enriched_df.columns
    assert "flight_number" in enriched_df.columns


def test_threshold_behavior(
    synthetic_pipeline_path: Path, valid_event_dict: dict[str, Any]
) -> None:
    """Test that decision threshold controls class assignment."""
    predictor_default = ModelPredictor(model_path=synthetic_pipeline_path, threshold=0.5)
    prob = predictor_default.predict_single(valid_event_dict).delay_probability

    # Threshold set below probability should predict delayed=1
    lower_threshold = max(0.0, prob - 0.05)
    predictor_low = ModelPredictor(model_path=synthetic_pipeline_path, threshold=lower_threshold)
    assert predictor_low.predict_single(valid_event_dict).predicted_delayed == 1

    # Threshold set above probability should predict delayed=0
    higher_threshold = min(1.0, prob + 0.05)
    predictor_high = ModelPredictor(model_path=synthetic_pipeline_path, threshold=higher_threshold)
    assert predictor_high.predict_single(valid_event_dict).predicted_delayed == 0


def test_invalid_threshold_raises_value_error(synthetic_pipeline_path: Path) -> None:
    """Ensure thresholds outside [0.0, 1.0] raise ValueError."""
    with pytest.raises(ValueError, match="Decision threshold must be between 0.0 and 1.0"):
        ModelPredictor(model_path=synthetic_pipeline_path, threshold=-0.1)

    with pytest.raises(ValueError, match="Decision threshold must be between 0.0 and 1.0"):
        ModelPredictor(model_path=synthetic_pipeline_path, threshold=1.05)


# -----------------------------------------------------------------------------
# 3. Feature Mapping, Order, and Schema Validation
# -----------------------------------------------------------------------------


def test_feature_order_independence(
    synthetic_pipeline_path: Path, valid_event_dict: dict[str, Any]
) -> None:
    """Ensure features provided in arbitrary column order are correctly aligned."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)

    df_natural = pd.DataFrame([valid_event_dict])[list(EXPECTED_FEATURE_COLUMNS)]
    # Reverse column order
    reversed_cols = list(EXPECTED_FEATURE_COLUMNS)[::-1]
    df_reversed = pd.DataFrame([valid_event_dict])[reversed_cols]

    res_natural = predictor.predict(df_natural)
    res_reversed = predictor.predict(df_reversed)

    assert res_natural[0].delay_probability == res_reversed[0].delay_probability
    assert res_natural[0].predicted_delayed == res_reversed[0].predicted_delayed


def test_missing_required_features_raises_error(
    synthetic_pipeline_path: Path, valid_event_dict: dict[str, Any]
) -> None:
    """Ensure missing any of the 15 features raises SchemaValidationError."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)

    # Drop distance
    bad_dict = dict(valid_event_dict)
    del bad_dict["distance"]
    with pytest.raises(SchemaValidationError, match="Missing required feature columns.*distance"):
        predictor.predict(bad_dict)

    # Drop carrier
    bad_dict2 = dict(valid_event_dict)
    del bad_dict2["carrier"]
    with pytest.raises(SchemaValidationError, match="Missing required feature columns.*carrier"):
        predictor.predict(bad_dict2)


def test_missing_or_nan_values_rejected(
    synthetic_pipeline_path: Path, valid_event_dict: dict[str, Any]
) -> None:
    """Ensure None, np.nan, or missing values in features are rejected."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)

    # NaN in numerical column
    bad_event = dict(valid_event_dict)
    bad_event["crs_elapsed_time"] = np.nan
    with pytest.raises(SchemaValidationError, match="Feature columns contain null or NaN values"):
        predictor.predict(bad_event)

    # None in categorical column
    bad_event2 = dict(valid_event_dict)
    bad_event2["origin"] = None
    with pytest.raises(SchemaValidationError, match="Feature columns contain null or NaN values"):
        predictor.predict(bad_event2)


@pytest.mark.parametrize(
    ("col_name", "invalid_val", "expected_err"),
    [
        ("crs_dep_hour", 24, "crs_dep_hour must be between 0 and 23"),
        ("crs_dep_hour", -1, "crs_dep_hour must be between 0 and 23"),
        ("crs_dep_minute", 60, "crs_dep_minute must be between 0 and 59"),
        ("crs_arr_hour", 25, "crs_arr_hour must be between 0 and 23"),
        ("crs_arr_minute", -1, "crs_arr_minute must be between 0 and 59"),
        ("month", 0, "month must be between 1 and 12"),
        ("month", 13, "month must be between 1 and 12"),
        ("day_of_month", 0, "day_of_month must be between 1 and 31"),
        ("day_of_month", 32, "day_of_month must be between 1 and 31"),
        ("day_of_week", 0, "day_of_week must be between 1 and 7"),
        ("day_of_week", 8, "day_of_week must be between 1 and 7"),
        ("is_weekend", 2, "is_weekend must be 0 or 1"),
        ("crs_elapsed_time", 0.0, "crs_elapsed_time must be strictly positive"),
        ("crs_elapsed_time", -15.0, "crs_elapsed_time must be strictly positive"),
        ("distance", 0.0, "distance must be strictly positive"),
        ("distance", -500.0, "distance must be strictly positive"),
        ("crs_elapsed_time", "non_numeric_str", "Non-numeric values found"),
        ("distance", np.inf, "Infinite values found"),
    ],
)
def test_invalid_numerical_bounds_rejected(
    synthetic_pipeline_path: Path,
    valid_event_dict: dict[str, Any],
    col_name: str,
    invalid_val: Any,
    expected_err: str,
) -> None:
    """Ensure out-of-bounds or invalid numerical values raise SchemaValidationError."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)
    bad_event = dict(valid_event_dict)
    bad_event[col_name] = invalid_val

    with pytest.raises(SchemaValidationError, match=expected_err):
        predictor.predict(bad_event)


def test_empty_categorical_string_rejected(
    synthetic_pipeline_path: Path, valid_event_dict: dict[str, Any]
) -> None:
    """Ensure blank or empty categorical features raise SchemaValidationError."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)
    bad_event = dict(valid_event_dict)
    bad_event["carrier"] = "   "

    with pytest.raises(SchemaValidationError, match="contains empty or blank strings"):
        predictor.predict(bad_event)


# -----------------------------------------------------------------------------
# 4. Target Leakage Rejection Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leakage_col",
    [
        "dep_delay",
        "arr_delay",
        "DEP_DELAY",
        "ARR_DELAY",
        "dep_time",
        "arr_time",
        "DEP_TIME",
        "ARR_TIME",
        "is_delayed_15",
        "actual_elapsed_time",
        "air_time",
        "taxi_in",
        "taxi_out",
        "wheels_off",
        "wheels_on",
        "cancellation_code",
        "cancelled",
        "diverted",
        "dep_del15",
        "actual_dep_time",
        "actual_arr_time",
    ],
)
def test_target_leakage_rejection(
    synthetic_pipeline_path: Path,
    valid_event_dict: dict[str, Any],
    leakage_col: str,
) -> None:
    """Ensure any prohibited target leakage column immediately raises TargetLeakageDetectedError."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)
    leakage_payload = dict(valid_event_dict)
    leakage_payload[leakage_col] = 10.0

    with pytest.raises(TargetLeakageDetectedError) as exc_info:
        predictor.predict(leakage_payload)

    assert "Target leakage columns detected" in str(exc_info.value)
    assert leakage_col.lower() in str(exc_info.value).lower()


# -----------------------------------------------------------------------------
# 5. Empty & Edge Case Inputs
# -----------------------------------------------------------------------------


def test_empty_dataframe_handling(synthetic_pipeline_path: Path) -> None:
    """Ensure empty DataFrames produce empty results gracefully without errors."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)
    empty_df = pd.DataFrame(columns=list(EXPECTED_FEATURE_COLUMNS))

    assert predictor.predict(empty_df) == []

    enriched = predictor.predict_dataframe(empty_df)
    assert len(enriched) == 0
    assert "delay_probability" in enriched.columns
    assert "predicted_delayed" in enriched.columns


def test_model_artifact_features_match_allowlist() -> None:
    """Regression test: verify trained model artifact feature names and order match allowlist."""
    from flight_delay.processing.feature_engineering import FEATURE_ALLOWLIST

    assert len(EXPECTED_FEATURE_COLUMNS) == 15
    assert EXPECTED_FEATURE_COLUMNS == tuple(FEATURE_ALLOWLIST)

    expected_features_list = [
        "carrier",
        "origin",
        "dest",
        "crs_dep_hour",
        "crs_dep_minute",
        "dep_time_blk",
        "crs_arr_hour",
        "crs_arr_minute",
        "arr_time_blk",
        "month",
        "day_of_month",
        "day_of_week",
        "is_weekend",
        "crs_elapsed_time",
        "distance",
    ]
    assert list(EXPECTED_FEATURE_COLUMNS) == expected_features_list

    assert DEFAULT_MODEL_PATH.exists() and DEFAULT_MODEL_PATH.is_file(), (
        f"Baseline model artifact missing at {DEFAULT_MODEL_PATH}"
    )
    artifact = joblib.load(DEFAULT_MODEL_PATH)
    model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
    assert hasattr(model, "feature_names_in_")
    assert list(model.feature_names_in_) == expected_features_list


def test_predict_dataframe_preserves_event_metadata_separately(
    synthetic_pipeline_path: Path, valid_event_dict: dict[str, Any]
) -> None:
    """Verify that inference extracts only 15 features while preserving metadata columns."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)
    df = pd.DataFrame([valid_event_dict])
    df["extra_metadata_field"] = "non_feature_metadata_value"
    df["event_timestamp"] = "2026-06-15T10:30:00Z"

    enriched = predictor.predict_dataframe(df)

    # 15 features plus metadata fields are preserved
    assert enriched["extra_metadata_field"].iloc[0] == "non_feature_metadata_value"
    assert enriched["event_timestamp"].iloc[0] == "2026-06-15T10:30:00Z"
    assert enriched["event_id"].iloc[0] == "test-event-001"
    assert enriched["flight_number"].iloc[0] == "100"

    # Prediction outputs added
    assert "delay_probability" in enriched.columns
    assert "predicted_delayed" in enriched.columns
    assert "prediction_time" in enriched.columns
    assert "model_version" in enriched.columns


def test_predict_preserves_existing_prediction_time(
    synthetic_pipeline_path: Path, valid_event_dict: dict[str, Any]
) -> None:
    """Ensure predict preserves existing prediction_time without generating a new one."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)
    custom_time = "2026-10-24T00:00:00+00:00"
    event_with_time = dict(valid_event_dict)
    event_with_time["prediction_time"] = custom_time

    results = predictor.predict(event_with_time)
    assert len(results) == 1
    assert results[0].prediction_time == custom_time


def test_predict_dataframe_preserves_existing_prediction_time(
    synthetic_pipeline_path: Path, valid_event_dict: dict[str, Any]
) -> None:
    """Ensure predict_dataframe preserves existing prediction_time column values."""
    predictor = ModelPredictor(model_path=synthetic_pipeline_path)
    custom_time = "2026-10-24T00:00:00+00:00"
    df = pd.DataFrame([valid_event_dict])
    df["prediction_time"] = custom_time

    enriched = predictor.predict_dataframe(df)
    assert enriched["prediction_time"].iloc[0] == custom_time
