"""Model inference module for real-time flight delay prediction.

Loads fitted baseline model artifacts, enforces feature schema and order contracts,
rejects target leakage fields, and computes delay probabilities and class predictions.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flight_delay.ingestion.contract import PROHIBITED_LEAKAGE_COLUMNS
from flight_delay.processing.feature_engineering import (
    FEATURE_ALLOWLIST,
    PROHIBITED_COLUMNS,
)

# Canonical 15-feature list in exact order expected by the trained baseline pipeline
EXPECTED_FEATURE_COLUMNS: tuple[str, ...] = tuple(FEATURE_ALLOWLIST)

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "carrier",
    "origin",
    "dest",
    "dep_time_blk",
    "arr_time_blk",
)

NUMERICAL_COLUMNS: tuple[str, ...] = (
    "crs_dep_hour",
    "crs_dep_minute",
    "crs_arr_hour",
    "crs_arr_minute",
    "month",
    "day_of_month",
    "day_of_week",
    "is_weekend",
    "crs_elapsed_time",
    "distance",
)

# Combined case-insensitive prohibited leakage fields
ALL_PROHIBITED_LEAKAGE_FIELDS: frozenset[str] = frozenset(
    {name.lower() for name in PROHIBITED_LEAKAGE_COLUMNS}
    | {
        name.lower()
        for name in PROHIBITED_COLUMNS
        if "delay" in name.lower() or "dep" in name.lower()
    }
    | {
        "dep_del15",
        "actual_dep_time",
        "actual_arr_time",
        "actual_dep",
        "actual_arr",
    }
)

DEFAULT_MODEL_PATH: Path = Path("models/baseline/model.joblib")


class InferenceError(RuntimeError):
    """Base exception for model inference failures."""


class ModelArtifactNotFoundError(InferenceError, FileNotFoundError):
    """Raised when the specified model artifact file does not exist."""


class ModelLoadError(InferenceError):
    """Raised when the model artifact file is corrupt or cannot be loaded."""


class SchemaValidationError(InferenceError, ValueError):
    """Raised when input features do not match expected schema, types, bounds, or contain NaNs."""


class TargetLeakageDetectedError(InferenceError, ValueError):
    """Raised when target leakage fields are detected in inference input."""


@dataclass(frozen=True)
class PredictionResult:
    """Structured result of a single flight delay prediction."""

    delay_probability: float
    predicted_delayed: int
    event_id: str | None = None
    prediction_time: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )
    model_version: str = "baseline-v1"

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction result to dictionary."""
        return asdict(self)


class ModelPredictor:
    """Reusable inference engine for real-time flight delay prediction."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        threshold: float = 0.5,
        model_version: str = "baseline-v1",
    ) -> None:
        """Initialize ModelPredictor by loading the fitted pipeline artifact.

        Args:
            model_path: Path to serialized model joblib file.
                Defaults to models/baseline/model.joblib.
            threshold: Decision threshold for delay classification in [0.0, 1.0]. Default 0.5.
            model_version: Version identifier string for predictions.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Decision threshold must be between 0.0 and 1.0, got {threshold}")

        self.threshold: float = threshold
        self.model_version: str = model_version
        self.model_path: Path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH

        self._pipeline: Any = self._load_model_artifact(self.model_path)
        self.feature_names: tuple[str, ...] = EXPECTED_FEATURE_COLUMNS

    @staticmethod
    def _load_model_artifact(path: Path) -> Any:
        """Load and validate the saved scikit-learn pipeline artifact."""
        if not path.exists() or not path.is_file():
            raise ModelArtifactNotFoundError(
                f"Model artifact not found at '{path}'. "
                "Ensure Phase 3 baseline training has executed and generated model.joblib."
            )

        try:
            model = joblib.load(path)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to deserialize model artifact from '{path}': {exc}"
            ) from exc

        if not hasattr(model, "predict_proba"):
            raise ModelLoadError(
                f"Loaded model of type '{type(model).__name__}' does not implement predict_proba."
            )

        return model

    def validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input DataFrame schema, reject leakage, check bounds, and return reordered X.

        Args:
            df: Input pandas DataFrame containing at least the 15 required feature columns.

        Returns:
            Cleaned and strictly ordered feature DataFrame X matching EXPECTED_FEATURE_COLUMNS.
        """
        if not isinstance(df, pd.DataFrame):
            raise SchemaValidationError(f"Expected pandas DataFrame, got {type(df).__name__}")

        # 1. Leakage check: inspect all columns in input DataFrame (case-insensitive)
        leakage_detected: list[str] = []
        for col in df.columns:
            col_str = str(col).strip().lower()
            if col_str in ALL_PROHIBITED_LEAKAGE_FIELDS:
                leakage_detected.append(str(col))
            elif "dep_delay" in col_str or "arr_delay" in col_str:
                leakage_detected.append(str(col))
            elif "actual_dep" in col_str or "actual_arr" in col_str:
                leakage_detected.append(str(col))

        if leakage_detected:
            raise TargetLeakageDetectedError(
                f"Target leakage columns detected in model input: {sorted(set(leakage_detected))}. "
                "Pre-departure model inference strictly prohibits actual post-departure fields."
            )

        # 2. Check for missing required feature columns
        missing_cols = [c for c in EXPECTED_FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            raise SchemaValidationError(
                f"Missing required feature columns for model inference: {missing_cols}"
            )

        if len(df) == 0:
            return pd.DataFrame(columns=list(EXPECTED_FEATURE_COLUMNS))

        # 3. Check for missing / NaN / None values in feature columns
        null_counts = df[list(EXPECTED_FEATURE_COLUMNS)].isna().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            raise SchemaValidationError(
                f"Feature columns contain null or NaN values: {cols_with_nulls.to_dict()}. "
                "Inference requires complete feature sets without silent imputation."
            )

        # 4. Validate numerical feature types, bounds, and finiteness
        for col_name in NUMERICAL_COLUMNS:
            series = pd.to_numeric(df[col_name], errors="coerce")
            if series.isna().any():
                raise SchemaValidationError(
                    f"Non-numeric values found in numerical feature '{col_name}'"
                )
            if not np.isfinite(series).all():
                raise SchemaValidationError(
                    f"Infinite values found in numerical feature '{col_name}'"
                )

        if not df["crs_dep_hour"].between(0, 23).all():
            raise SchemaValidationError("crs_dep_hour must be between 0 and 23")
        if not df["crs_dep_minute"].between(0, 59).all():
            raise SchemaValidationError("crs_dep_minute must be between 0 and 59")
        if not df["crs_arr_hour"].between(0, 23).all():
            raise SchemaValidationError("crs_arr_hour must be between 0 and 23")
        if not df["crs_arr_minute"].between(0, 59).all():
            raise SchemaValidationError("crs_arr_minute must be between 0 and 59")
        if not df["month"].between(1, 12).all():
            raise SchemaValidationError("month must be between 1 and 12")
        if not df["day_of_month"].between(1, 31).all():
            raise SchemaValidationError("day_of_month must be between 1 and 31")
        if not df["day_of_week"].between(1, 7).all():
            raise SchemaValidationError("day_of_week must be between 1 and 7")
        if not df["is_weekend"].isin([0, 1]).all():
            raise SchemaValidationError("is_weekend must be 0 or 1")
        if (df["crs_elapsed_time"] <= 0.0).any():
            raise SchemaValidationError("crs_elapsed_time must be strictly positive")
        if (df["distance"] <= 0.0).any():
            raise SchemaValidationError("distance must be strictly positive")

        # 5. Validate categorical features
        for col_name in CATEGORICAL_COLUMNS:
            str_series = df[col_name].astype(str).str.strip()
            if str_series.eq("").any() or str_series.str.lower().eq("none").any():
                raise SchemaValidationError(
                    f"Categorical feature '{col_name}' contains empty or blank strings"
                )

        # 6. Extract and order feature subset exactly
        X = df[list(EXPECTED_FEATURE_COLUMNS)].copy()
        for num_col in NUMERICAL_COLUMNS:
            X[num_col] = pd.to_numeric(X[num_col])
        for cat_col in CATEGORICAL_COLUMNS:
            X[cat_col] = X[cat_col].astype(str)

        return X

    def predict(
        self,
        data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    ) -> list[PredictionResult]:
        """Compute delay predictions and return a list of structured PredictionResults.

        Args:
            data: DataFrame, single dict record, or list of dict records containing features.

        Returns:
            List of PredictionResult dataclasses containing delay probabilities and classes.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise SchemaValidationError(f"Unsupported input data type: {type(data).__name__}")

        if len(df) == 0:
            return []

        X = self.validate_features(df)

        try:
            probs = self._pipeline.predict_proba(X)
        except Exception as exc:
            raise InferenceError(
                f"Model prediction failed during pipeline evaluation: {exc}"
            ) from exc

        # Extract delay probabilities (column index 1)
        delay_probs = probs[:, 1]
        if (
            not np.isfinite(delay_probs).all()
            or not ((delay_probs >= 0.0) & (delay_probs <= 1.0)).all()
        ):
            raise InferenceError(
                "Model generated invalid probabilities (NaN, Inf, or out-of-range)."
            )

        predicted_delayed = (delay_probs >= self.threshold).astype(int)
        now_utc = datetime.datetime.now(datetime.UTC).isoformat()

        pred_times: list[str]
        if "prediction_time" in df.columns:
            pred_times = [
                str(t) if pd.notna(t) and str(t).strip() else now_utc for t in df["prediction_time"]
            ]
        else:
            pred_times = [now_utc] * len(df)

        event_ids: list[str | None] = (
            df["event_id"].astype(str).tolist() if "event_id" in df.columns else [None] * len(df)
        )

        results: list[PredictionResult] = []
        for i in range(len(df)):
            results.append(
                PredictionResult(
                    delay_probability=float(round(delay_probs[i], 6)),
                    predicted_delayed=int(predicted_delayed[i]),
                    event_id=event_ids[i],
                    prediction_time=pred_times[i],
                    model_version=self.model_version,
                )
            )

        return results

    def predict_single(self, record: dict[str, Any] | pd.Series) -> PredictionResult:
        """Compute delay prediction for a single record."""
        if isinstance(record, pd.Series):
            rec_dict = record.to_dict()
        elif isinstance(record, dict):
            rec_dict = record
        else:
            raise SchemaValidationError(f"Expected dict or pd.Series, got {type(record).__name__}")

        results = self.predict(rec_dict)
        if not results:
            raise InferenceError("Failed to generate prediction result for record.")
        return results[0]

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute predictions and append prediction columns to the input DataFrame.

        Appended columns:
        - `delay_probability`: float64
        - `predicted_delayed`: int32
        - `prediction_time`: string (ISO 8601 UTC)
        - `model_version`: string

        Args:
            df: Input DataFrame containing required features.

        Returns:
            New DataFrame with input columns and attached prediction metadata.
        """
        if len(df) == 0:
            res_df = df.copy()
            res_df["delay_probability"] = pd.Series(dtype=float)
            res_df["predicted_delayed"] = pd.Series(dtype=int)
            res_df["prediction_time"] = pd.Series(dtype=str)
            res_df["model_version"] = pd.Series(dtype=str)
            return res_df

        X = self.validate_features(df)

        try:
            probs = self._pipeline.predict_proba(X)
        except Exception as exc:
            raise InferenceError(
                f"Model prediction failed during pipeline evaluation: {exc}"
            ) from exc

        delay_probs = probs[:, 1]
        if (
            not np.isfinite(delay_probs).all()
            or not ((delay_probs >= 0.0) & (delay_probs <= 1.0)).all()
        ):
            raise InferenceError(
                "Model generated invalid probabilities (NaN, Inf, or out-of-range)."
            )

        predicted_delayed = (delay_probs >= self.threshold).astype(int)
        now_utc = datetime.datetime.now(datetime.UTC).isoformat()

        output_df = df.copy()
        output_df["delay_probability"] = np.round(delay_probs, 6)
        output_df["predicted_delayed"] = predicted_delayed
        if "prediction_time" not in output_df.columns:
            output_df["prediction_time"] = now_utc
        else:
            output_df["prediction_time"] = output_df["prediction_time"].fillna(now_utc)
        output_df["model_version"] = self.model_version

        return output_df
