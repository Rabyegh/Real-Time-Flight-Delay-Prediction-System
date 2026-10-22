"""Phase 3 baseline model training, evaluation, and artifact generation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.parse
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flight_delay.processing.feature_engineering import (
    FEATURE_ALLOWLIST,
    PROHIBITED_COLUMNS,
    TARGET_COLUMN,
)

LOGGER = logging.getLogger(__name__)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "carrier",
    "origin",
    "dest",
    "dep_time_blk",
    "arr_time_blk",
)

NUMERICAL_FEATURES: tuple[str, ...] = (
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

# Verify feature consistency at import time
assert set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES) == set(FEATURE_ALLOWLIST), (
    "Categorical and numerical feature definitions must exactly match FEATURE_ALLOWLIST."
)


class ModelTrainingError(RuntimeError):
    """Base exception for model training and evaluation failures."""


class SchemaValidationError(ModelTrainingError):
    """Raised when input feature dataset does not meet schema requirements."""


class DataSplitError(ModelTrainingError):
    """Raised when data splitting fails or results in empty/invalid subsets."""


class PathSafetyError(ModelTrainingError):
    """Raised when output artifact directory collides with or nests inside input dataset."""


class MLflowTrackingError(ModelTrainingError):
    """Raised when MLflow tracking URI, experiment, or run configuration fails."""


def validate_tracking_uri(tracking_uri: str | None) -> str:
    """Validate and resolve MLflow tracking URI."""
    if tracking_uri is not None:
        trimmed = tracking_uri.strip()
        if not trimmed:
            raise MLflowTrackingError("MLflow tracking URI cannot be empty or whitespace.")
        uri = trimmed
    else:
        env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        uri = env_uri if env_uri else "file:./mlruns"

    parsed = urllib.parse.urlparse(uri)
    valid_schemes = {"", "file", "http", "https", "sqlite", "postgresql", "mysql", "mssql"}
    if parsed.scheme and parsed.scheme.lower() not in valid_schemes:
        raise MLflowTrackingError(
            f"Unsupported or invalid MLflow tracking URI scheme '{parsed.scheme}': {uri}"
        )
    return uri


def validate_experiment_name(experiment_name: str | None) -> str:
    """Validate and resolve MLflow experiment name."""
    if experiment_name is None:
        return "flight-delay-baseline"
    name = experiment_name.strip()
    if not name:
        raise MLflowTrackingError("MLflow experiment name cannot be empty or whitespace.")
    return name


def validate_training_paths(input_path: Path | str, output_dir: Path | str) -> tuple[Path, Path]:
    """Validate input exists and output directory does not collide with or reside inside input."""
    in_path = Path(input_path)
    out_path = Path(output_dir)
    if not in_path.exists():
        raise SchemaValidationError(f"Input path does not exist: {input_path}")

    input_resolved = in_path.resolve()
    output_resolved = out_path.resolve()

    if input_resolved == output_resolved:
        raise PathSafetyError(
            f"Output directory cannot be identical to input dataset path: {output_dir}"
        )
    if input_resolved in output_resolved.parents:
        raise PathSafetyError(
            f"Output directory cannot be nested inside input dataset directory: {output_dir}"
        )
    if output_resolved in input_resolved.parents:
        raise PathSafetyError(
            f"Output directory cannot contain input dataset directory: {output_dir}"
        )

    return input_resolved, output_resolved


def validate_split_config(config: SplitConfig) -> None:
    """Validate chronological consistency and boundary bounds for data splitting."""
    if not (1 <= config.train_june_end_day <= 31):
        raise DataSplitError(
            f"Invalid train_june_end_day: {config.train_june_end_day}. Must be between 1 and 31."
        )
    if not (1 <= config.val_start_day <= config.val_end_day <= 31):
        raise DataSplitError(
            f"Invalid validation day range: [{config.val_start_day}, {config.val_end_day}]."
        )
    if not (1 <= config.test_start_day <= config.test_end_day <= 31):
        raise DataSplitError(
            f"Invalid test day range: [{config.test_start_day}, {config.test_end_day}]."
        )

    # Check overlaps when val and test share the same month
    if config.val_month == config.test_month and config.val_end_day >= config.test_start_day:
        raise DataSplitError(
            f"Validation and test boundaries overlap or are inverted in month {config.val_month}: "
            f"val ends at day {config.val_end_day}, test starts at day {config.test_start_day}."
        )

    # Check overlaps between train and val
    if (
        config.val_month in config.train_months
        and config.train_june_end_day >= config.val_start_day
    ):
        raise DataSplitError(
            f"Train and validation boundaries overlap in month {config.val_month}: "
            f"train ends at day {config.train_june_end_day}, "
            f"val starts at day {config.val_start_day}."
        )


@dataclass(frozen=True)
class SplitConfig:
    """Chronological boundary configuration for train, validation, and test sets.

    Default boundaries for 2023 February and June data:
    - Train: All February (Feb 1-28) + June 1 to June 16 (44 days)
    - Validation: June 17 to June 23 (7 full days, Saturday to Friday)
    - Test: June 24 to June 30 (7 full days, Saturday to Friday)
    """

    train_months: tuple[int, ...] = (2, 6)
    train_june_end_day: int = 16
    val_month: int = 6
    val_start_day: int = 17
    val_end_day: int = 23
    test_month: int = 6
    test_start_day: int = 24
    test_end_day: int = 30


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for baseline training, evaluation, and experiment tracking."""

    input_path: Path = Path("data/features/dataset")
    output_dir: Path = Path("models/baseline")
    decision_threshold: float = 0.5
    class_weight: str | None = None
    max_iter: int = 500
    random_state: int = 42
    split_config: SplitConfig = field(default_factory=SplitConfig)
    tracking_uri: str | None = None
    experiment_name: str = "flight-delay-baseline"
    run_name: str | None = None
    enable_mlflow: bool = True


@dataclass
class SplitMetadata:
    """Detailed metadata for a dataset split."""

    train_dates: str
    val_dates: str
    test_dates: str
    train_rows: int
    val_rows: int
    test_rows: int
    total_rows: int
    train_positives: int
    val_positives: int
    test_positives: int
    train_prevalence: float
    val_prevalence: float
    test_prevalence: float
    dates_disjoint: bool


@dataclass
class DatasetSplit:
    """Container for partitioned train, validation, and test datasets."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    metadata: SplitMetadata


def assert_feature_columns(columns: Sequence[str], allow_target: bool = False) -> None:
    """Validate columns contain all allowlisted features and no unapproved columns."""
    col_set = set(columns)
    missing = [c for c in FEATURE_ALLOWLIST if c not in col_set]
    if missing:
        raise SchemaValidationError(f"Missing required feature columns: {missing}")

    allowed = set(FEATURE_ALLOWLIST)
    if allow_target:
        allowed.add(TARGET_COLUMN)

    unexpected = sorted(col_set - allowed)
    if unexpected:
        raise SchemaValidationError(
            f"Unexpected unapproved columns present in feature dataset: {unexpected}. "
            f"Only the approved {len(FEATURE_ALLOWLIST)} features are permitted."
        )

    prohibited_upper = {name.upper() for name in PROHIBITED_COLUMNS}
    for col in columns:
        if col.upper() in prohibited_upper:
            raise SchemaValidationError(f"Prohibited leakage column present in dataset: {col}")
        if col.upper() == "DEP_TIME" or "ACTUAL_DEP" in col.upper():
            raise SchemaValidationError(f"Post-departure actual time field detected: {col}")
        if "DEP_DELAY" in col.upper() or "ARR_DELAY" in col.upper():
            raise SchemaValidationError(f"Delay outcome field detected in feature columns: {col}")


def load_feature_dataset(dataset_path: Path | str) -> pd.DataFrame:
    """Load and validate the feature dataset Parquet files."""
    path = Path(dataset_path)
    if not path.exists():
        raise SchemaValidationError(f"Feature dataset path does not exist: {path}")

    try:
        df = pd.read_parquet(str(path))
    except Exception as exc:
        raise SchemaValidationError(f"Failed to read Parquet dataset from {path}: {exc}") from exc

    if len(df) == 0:
        raise SchemaValidationError(f"Feature dataset at {path} is empty (0 rows).")

    # Validate target presence and values
    if TARGET_COLUMN not in df.columns:
        raise SchemaValidationError(f"Target column '{TARGET_COLUMN}' is missing from dataset.")

    target_vals = df[TARGET_COLUMN].dropna().unique()
    if not set(target_vals).issubset({0, 1}):
        raise SchemaValidationError(
            f"Target column '{TARGET_COLUMN}' must contain only binary values 0 and 1, "
            f"got {target_vals}."
        )

    if df[TARGET_COLUMN].isnull().any():
        raise SchemaValidationError(f"Target column '{TARGET_COLUMN}' contains null values.")

    # Validate feature columns and leakage (excluding TARGET_COLUMN)
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    assert_feature_columns(feature_cols, allow_target=False)

    # Ensure month is integer if stored as category from Hive partitioning
    if "month" in df.columns:
        df["month"] = df["month"].astype(int)

    # Check for missing values in features
    missing_counts = df[list(FEATURE_ALLOWLIST)].isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if not cols_with_missing.empty:
        raise SchemaValidationError(
            f"Feature columns contain null values: {cols_with_missing.to_dict()}"
        )

    return df


def split_dataset_chronological(
    df: pd.DataFrame,
    config: SplitConfig | None = None,
) -> DatasetSplit:
    """Partition the dataset into chronological train, validation, and test subsets.

    Guarantees:
    - Strict chronological boundary separation.
    - Zero date overlap between subsets (same calendar date cannot appear in multiple splits).
    - Validation and Test subsets cover full 7-day calendar weeks (Saturday to Friday).
    - Every row is assigned to exactly one subset; unassigned rows raise DataSplitError.
    """
    config = config or SplitConfig()
    validate_split_config(config)

    month = df["month"].astype(int)
    day = df["day_of_month"].astype(int)

    # Pure train months: months in train_months that are entirely allocated to training
    pure_train_months = [m for m in config.train_months if m != config.val_month]

    # Build disjoint date masks respecting train_months configuration
    train_mask = (month.isin(pure_train_months)) | (
        (month == config.val_month) & (day <= config.train_june_end_day)
    )
    val_mask = (
        (month == config.val_month) & (day >= config.val_start_day) & (day <= config.val_end_day)
    )
    test_mask = (
        (month == config.test_month) & (day >= config.test_start_day) & (day <= config.test_end_day)
    )

    # Verify date disjointness
    overlap_train_val = (train_mask & val_mask).any()
    overlap_train_test = (train_mask & test_mask).any()
    overlap_val_test = (val_mask & test_mask).any()

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise DataSplitError("Data split boundaries overlap; date disjointness violated.")

    # Verify every row belongs to exactly one subset
    assigned_mask = train_mask | val_mask | test_mask
    unassigned_count = int((~assigned_mask).sum())
    if unassigned_count > 0:
        unassigned_sample = (
            df.loc[~assigned_mask, ["month", "day_of_month"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise DataSplitError(
            f"{unassigned_count} rows in input dataset fall outside configured chronological "
            f"split boundaries. Sample unassigned dates: {unassigned_sample[:5]}."
        )

    train_df = df.loc[train_mask]
    val_df = df.loc[val_mask]
    test_df = df.loc[test_mask]

    if len(train_df) == 0:
        raise DataSplitError("Training split is empty.")
    if len(val_df) == 0:
        raise DataSplitError("Validation split is empty.")
    if len(test_df) == 0:
        raise DataSplitError("Test split is empty.")

    feature_cols = list(FEATURE_ALLOWLIST)
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN].astype(int)
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN].astype(int)
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN].astype(int)

    start_month = min(config.train_months) if config.train_months else 2
    metadata = SplitMetadata(
        train_dates=f"2023-{start_month:02d}-01 to 2023-06-{config.train_june_end_day:02d}",
        val_dates=f"2023-06-{config.val_start_day:02d} to 2023-06-{config.val_end_day:02d}",
        test_dates=f"2023-06-{config.test_start_day:02d} to 2023-06-{config.test_end_day:02d}",
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        total_rows=len(train_df) + len(val_df) + len(test_df),
        train_positives=int(y_train.sum()),
        val_positives=int(y_val.sum()),
        test_positives=int(y_test.sum()),
        train_prevalence=round(float(y_train.mean()), 6),
        val_prevalence=round(float(y_val.mean()), 6),
        test_prevalence=round(float(y_test.mean()), 6),
        dates_disjoint=True,
    )

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        metadata=metadata,
    )


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build preprocessor: OneHotEncoder for categoricals and StandardScaler for numericals."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "num",
                StandardScaler(),
                list(NUMERICAL_FEATURES),
            ),
        ]
    )


def build_majority_baseline() -> DummyClassifier:
    """Build majority-class baseline classifier predicting the most frequent training class."""
    return DummyClassifier(strategy="most_frequent")


def build_logistic_regression_pipeline(
    class_weight: str | None = None,
    max_iter: int = 500,
    random_state: int = 42,
) -> Pipeline:
    """Build a complete scikit-learn pipeline for logistic regression."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute comprehensive classification evaluation metrics."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    total_samples = len(y_true_arr)
    positives = int(np.sum(y_true_arr == 1))
    negatives = int(np.sum(y_true_arr == 0))
    prevalence = round(positives / total_samples, 6) if total_samples > 0 else 0.0

    acc = round(float(accuracy_score(y_true_arr, y_pred_arr)), 6)
    prec = round(float(precision_score(y_true_arr, y_pred_arr, zero_division=0)), 6)
    rec = round(float(recall_score(y_true_arr, y_pred_arr, zero_division=0)), 6)
    f1 = round(float(f1_score(y_true_arr, y_pred_arr, zero_division=0)), 6)

    # ROC-AUC and PR-AUC require probabilities and at least one sample per class
    has_both_classes = len(np.unique(y_true_arr)) > 1
    if y_prob is not None and has_both_classes:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        roc_auc = round(float(roc_auc_score(y_true_arr, y_prob_arr)), 6)
        pr_auc = round(float(average_precision_score(y_true_arr, y_prob_arr)), 6)
    elif y_prob is not None:
        roc_auc = 0.5
        pr_auc = prevalence
    else:
        # For Dummy without probabilities
        roc_auc = 0.5
        pr_auc = prevalence

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "positive_prevalence": prevalence,
        "decision_threshold": threshold,
        "total_samples": total_samples,
        "positive_count": positives,
        "negative_count": negatives,
        "confusion_matrix": {
            "true_negative": cm[0][0],
            "false_positive": cm[0][1],
            "false_negative": cm[1][0],
            "true_positive": cm[1][1],
            "matrix_2x2": cm,
        },
    }


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a trained model or pipeline on a given feature set and target."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
    else:
        preds = model.predict(X)
        probs = None

    return evaluate_predictions(y, preds, y_prob=probs, threshold=threshold)


def _generate_markdown_report(metrics: dict[str, Any]) -> str:
    """Generate a clean markdown report summarizing the training and evaluation results."""
    split = metrics["split_metadata"]
    lr_test = metrics["models"]["logistic_regression"]["evaluation"]["test"]
    dummy_test = metrics["models"]["majority_baseline"]["evaluation"]["test"]
    lr_val = metrics["models"]["logistic_regression"]["evaluation"]["validation"]
    dummy_val = metrics["models"]["majority_baseline"]["evaluation"]["validation"]
    cm_test = lr_test["confusion_matrix"]
    cm_val = lr_val["confusion_matrix"]

    lines = [
        "# Baseline Model Evaluation Report",
        "",
        f"**Generated:** {metrics['timestamp']}",
        f"**Target:** `{metrics['target_column']}` (delay >= 15m)",
        f"**Decision Threshold:** {metrics['decision_threshold']}",
        "",
        "---",
        "",
        "## 1. Chronological Split Strategy & Data Distribution",
        "",
        "Because only February and June 2023 historical data are currently available,",
        "continuous time-series cross-validation across all seasons cannot be performed.",
        "A strict chronological cut was implemented to isolate future evaluation data from past",
        "training data while avoiding weekday bias:",
        "",
        f"- **Train:** {split['train_dates']} ({split['train_rows']:,} rows, "
        f"positive rate: {split['train_prevalence']:.2%})",
        "  - Covers all 28 days of February 2023 (winter) plus June 1-16 (early summer).",
        f"- **Validation:** {split['val_dates']} ({split['val_rows']:,} rows, "
        f"positive rate: {split['val_prevalence']:.2%})",
        "  - Covers exactly one full 7-day cycle (Saturday to Friday) in mid-June.",
        f"- **Test:** {split['test_dates']} ({split['test_rows']:,} rows, "
        f"positive rate: {split['test_prevalence']:.2%})",
        "  - Covers exactly one full 7-day cycle (Saturday to Friday) in late June.",
        "",
        "**Date Disjointness:** Strictly enforced. No single calendar date is shared across train,",
        "validation, or test sets. All preprocessing transformers are fitted exclusively on the",
        "training subset.",
        "",
        "---",
        "",
        "## 2. Evaluation Across Partitions",
        "",
        "> [!NOTE]",
        "> **Role of Validation Partition:** The validation subset (June 17–23) provides an",
        "> intermediate temporal benchmark to monitor seasonal performance shift before final",
        "> evaluation. In this baseline phase, the held-out test set remains strictly unaccessed",
        "> during model training, tuning, and selection. The validation partition is reported here",
        "> for temporal comparison and is reserved for future hyperparameter sweeps (e.g.,",
        "> regularization tuning, alternative algorithms) and decision-threshold calibration.",
        "> No threshold or hyperparameter tuning was applied in this baseline phase.",
        "",
        f"### Test Set Evaluation Comparison (Late June, N = {lr_test['total_samples']:,})",
        "",
        "| Metric | Majority Baseline (Predict 0) | Logistic Regression | Absolute Delta |",
        "|---|---|---|---|",
        f"| **Accuracy** | {dummy_test['accuracy']:.4f} | {lr_test['accuracy']:.4f} | "
        f"{lr_test['accuracy'] - dummy_test['accuracy']:+.4f} |",
        f"| **Precision** | {dummy_test['precision']:.4f} | {lr_test['precision']:.4f} | "
        f"{lr_test['precision'] - dummy_test['precision']:+.4f} |",
        f"| **Recall** | {dummy_test['recall']:.4f} | {lr_test['recall']:.4f} | "
        f"{lr_test['recall'] - dummy_test['recall']:+.4f} |",
        f"| **F1-Score** | {dummy_test['f1_score']:.4f} | {lr_test['f1_score']:.4f} | "
        f"{lr_test['f1_score'] - dummy_test['f1_score']:+.4f} |",
        f"| **ROC-AUC** | {dummy_test['roc_auc']:.4f} | {lr_test['roc_auc']:.4f} | "
        f"{lr_test['roc_auc'] - dummy_test['roc_auc']:+.4f} |",
        f"| **PR-AUC (Avg Precision)** | {dummy_test['pr_auc']:.4f} | {lr_test['pr_auc']:.4f} | "
        f"{lr_test['pr_auc'] - dummy_test['pr_auc']:+.4f} |",
        f"| **Positive Prevalence** | {dummy_test['positive_prevalence']:.4f} | "
        f"{lr_test['positive_prevalence']:.4f} | 0.0000 |",
        "",
        f"### Validation Set Evaluation Comparison (Mid-June, N = {lr_val['total_samples']:,})",
        "",
        "| Metric | Majority Baseline (Predict 0) | Logistic Regression | Absolute Delta |",
        "|---|---|---|---|",
        f"| **Accuracy** | {dummy_val['accuracy']:.4f} | {lr_val['accuracy']:.4f} | "
        f"{lr_val['accuracy'] - dummy_val['accuracy']:+.4f} |",
        f"| **Precision** | {dummy_val['precision']:.4f} | {lr_val['precision']:.4f} | "
        f"{lr_val['precision'] - dummy_val['precision']:+.4f} |",
        f"| **Recall** | {dummy_val['recall']:.4f} | {lr_val['recall']:.4f} | "
        f"{lr_val['recall'] - dummy_val['recall']:+.4f} |",
        f"| **F1-Score** | {dummy_val['f1_score']:.4f} | {lr_val['f1_score']:.4f} | "
        f"{lr_val['f1_score'] - dummy_val['f1_score']:+.4f} |",
        f"| **ROC-AUC** | {dummy_val['roc_auc']:.4f} | {lr_val['roc_auc']:.4f} | "
        f"{lr_val['roc_auc'] - dummy_val['roc_auc']:+.4f} |",
        f"| **PR-AUC (Avg Precision)** | {dummy_val['pr_auc']:.4f} | {lr_val['pr_auc']:.4f} | "
        f"{lr_val['pr_auc'] - dummy_val['pr_auc']:+.4f} |",
        f"| **Positive Prevalence** | {dummy_val['positive_prevalence']:.4f} | "
        f"{lr_val['positive_prevalence']:.4f} | 0.0000 |",
        "",
        "### Confusion Matrices",
        "",
        f"#### Test Set (Held-Out Late June, N = {lr_test['total_samples']:,})",
        "",
        f"- **True Negatives (TN):** {cm_test['true_negative']:,}",
        f"- **False Positives (FP):** {cm_test['false_positive']:,}",
        f"- **False Negatives (FN):** {cm_test['false_negative']:,}",
        f"- **True Positives (TP):** {cm_test['true_positive']:,}",
        "",
        f"#### Validation Set (Mid-June, N = {lr_val['total_samples']:,})",
        "",
        f"- **True Negatives (TN):** {cm_val['true_negative']:,}",
        f"- **False Positives (FP):** {cm_val['false_positive']:,}",
        f"- **False Negatives (FN):** {cm_val['false_negative']:,}",
        f"- **True Positives (TP):** {cm_val['true_positive']:,}",
        "",
        "---",
        "",
        "## 3. Why Accuracy Alone Is Insufficient",
        "",
        f"In an imbalanced classification problem with "
        f"{dummy_test['positive_prevalence']:.1%} delay:",
        f"- The naive majority-class baseline achieves **{dummy_test['accuracy']:.1%} accuracy**",
        "  simply by predicting that every single flight departs on time.",
        "- However, precision, recall, and F1-score are **0.0000**, and ROC-AUC is **0.5000**.",
        "- A model evaluated solely on accuracy rewards a system that never alerts on delays.",
        f"- Logistic Regression achieves ROC-AUC of {lr_test['roc_auc']:.4f} and PR-AUC of",
        f"  {lr_test['pr_auc']:.4f} (vs {dummy_test['positive_prevalence']:.4f} baseline),",
        f"  identifying {cm_test['true_positive']:,} delayed flights on held-out test data.",
        "",
        "---",
        "",
        "## 4. Limitations & Non-Production Notice",
        "",
        "> [!WARNING]",
        "> This model is a **baseline benchmark** and is **NOT production-ready**.",
        "",
        "1. **Non-Consecutive Months:** Historical dataset contains only February and June 2023",
        "   with a 3-month gap between them.",
        "2. **Seasonal Drift:** Delay rates shift from ~18.5% in winter to ~32.3% in peak",
        "   summer travel (late June). A model trained on winter data suffers from concept drift.",
        "3. **No Cross-Year Validation:** Performance on 2023 dates does not guarantee",
        "   generalization to holidays, severe weather events, or future operating years.",
        "4. **Pre-Departure Horizon:** Features are restricted to pre-departure schedules.",
        "   Live weather forecasts and upstream inbound delays are necessary for deployment.",
        "",
    ]
    return "\n".join(lines)


def _log_mlflow_run(
    config: TrainingConfig,
    split: DatasetSplit,
    lr_pipeline: Pipeline,
    dummy: DummyClassifier,
    metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Log parameters, metrics, tags, and model artifacts to MLflow."""
    import datetime

    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    tracking_uri = validate_tracking_uri(config.tracking_uri)
    experiment_name = validate_experiment_name(config.experiment_name)

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        raise MLflowTrackingError(
            f"Failed to initialize MLflow experiment '{experiment_name}' at '{tracking_uri}': {exc}"
        ) from exc

    if mlflow.active_run():
        mlflow.end_run()

    timestamp_str = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or f"logistic-regression-baseline-{timestamp_str}"

    lr_eval_train = metrics["models"]["logistic_regression"]["evaluation"]["train"]
    lr_eval_val = metrics["models"]["logistic_regression"]["evaluation"]["validation"]
    lr_eval_test = metrics["models"]["logistic_regression"]["evaluation"]["test"]
    dummy_eval_test = metrics["models"]["majority_baseline"]["evaluation"]["test"]
    cmp = metrics["comparison_summary"]

    with mlflow.start_run(run_name=run_name) as run:
        # 1. Tags
        mlflow.set_tags(
            {
                "model_family": "baseline",
                "algorithm": "logistic_regression",
                "target_column": TARGET_COLUMN,
                "temporal_scope": "2023-02,2023-06",
                "split_strategy": "chronological_date_disjoint",
                "validation_role": "temporal_monitoring_and_tuning_reserve",
                "test_role": "isolated_final_evaluation",
                "production_ready": "false",
            }
        )

        # 2. Parameters
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "solver": "lbfgs",
                "max_iter": config.max_iter,
                "random_state": config.random_state,
                "class_weight": str(config.class_weight),
                "decision_threshold": config.decision_threshold,
                "target_column": TARGET_COLUMN,
                "delay_threshold_minutes": 15.0,
                "feature_count": len(FEATURE_ALLOWLIST),
                "categorical_features": ",".join(CATEGORICAL_FEATURES),
                "numerical_features": ",".join(NUMERICAL_FEATURES),
                "train_dates": split.metadata.train_dates,
                "val_dates": split.metadata.val_dates,
                "test_dates": split.metadata.test_dates,
                "train_rows": split.metadata.train_rows,
                "val_rows": split.metadata.val_rows,
                "test_rows": split.metadata.test_rows,
                "total_rows": split.metadata.total_rows,
                "dates_disjoint": str(split.metadata.dates_disjoint),
            }
        )

        # 3. Metrics (Train, Validation, and Test separated)
        mlflow.log_metrics(
            {
                "train_accuracy": lr_eval_train["accuracy"],
                "train_precision": lr_eval_train["precision"],
                "train_recall": lr_eval_train["recall"],
                "train_f1_score": lr_eval_train["f1_score"],
                "train_roc_auc": lr_eval_train["roc_auc"],
                "train_pr_auc": lr_eval_train["pr_auc"],
                "train_positive_prevalence": lr_eval_train["positive_prevalence"],
                "val_accuracy": lr_eval_val["accuracy"],
                "val_precision": lr_eval_val["precision"],
                "val_recall": lr_eval_val["recall"],
                "val_f1_score": lr_eval_val["f1_score"],
                "val_roc_auc": lr_eval_val["roc_auc"],
                "val_pr_auc": lr_eval_val["pr_auc"],
                "val_positive_prevalence": lr_eval_val["positive_prevalence"],
                "val_true_negative": lr_eval_val["confusion_matrix"]["true_negative"],
                "val_false_positive": lr_eval_val["confusion_matrix"]["false_positive"],
                "val_false_negative": lr_eval_val["confusion_matrix"]["false_negative"],
                "val_true_positive": lr_eval_val["confusion_matrix"]["true_positive"],
                "test_accuracy": lr_eval_test["accuracy"],
                "test_precision": lr_eval_test["precision"],
                "test_recall": lr_eval_test["recall"],
                "test_f1_score": lr_eval_test["f1_score"],
                "test_roc_auc": lr_eval_test["roc_auc"],
                "test_pr_auc": lr_eval_test["pr_auc"],
                "test_positive_prevalence": lr_eval_test["positive_prevalence"],
                "test_true_negative": lr_eval_test["confusion_matrix"]["true_negative"],
                "test_false_positive": lr_eval_test["confusion_matrix"]["false_positive"],
                "test_false_negative": lr_eval_test["confusion_matrix"]["false_negative"],
                "test_true_positive": lr_eval_test["confusion_matrix"]["true_positive"],
                "test_accuracy_vs_dummy": cmp["test_accuracy_vs_dummy"],
                "test_roc_auc_vs_dummy": cmp["test_roc_auc_vs_dummy"],
                "test_pr_auc_vs_dummy": cmp["test_pr_auc_vs_dummy"],
                "test_f1_vs_dummy": cmp["test_f1_vs_dummy"],
                "dummy_test_accuracy": dummy_eval_test["accuracy"],
                "dummy_test_roc_auc": dummy_eval_test["roc_auc"],
                "dummy_test_pr_auc": dummy_eval_test["pr_auc"],
            }
        )

        # 4. Model Artifact with Schema Signature
        sample_input = split.X_test.head(5)
        signature = infer_signature(sample_input, lr_pipeline.predict(sample_input))
        mlflow.sklearn.log_model(
            sk_model=lr_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=sample_input,
        )

        # 5. File Artifacts
        for artifact_name in [
            "model.joblib",
            "dummy_model.joblib",
            "evaluation_metrics.json",
            "split_metadata.json",
            "evaluation_report.md",
        ]:
            file_path = output_dir / artifact_name
            if file_path.is_file():
                mlflow.log_artifact(str(file_path), artifact_path="artifacts")

        run_info = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "tracking_uri": tracking_uri,
            "run_name": run_name,
        }
        LOGGER.info(
            "MLflow tracking completed: run_id=%s, experiment=%s at %s",
            run_info["run_id"],
            experiment_name,
            tracking_uri,
        )
        return run_info


def train_and_evaluate(config: TrainingConfig | None = None) -> dict[str, Any]:
    """Execute the full training, validation, evaluation, and artifact saving pipeline."""
    import datetime

    config = config or TrainingConfig()
    LOGGER.info("Starting baseline training pipeline from %s", config.input_path)

    # 0. Safety validation of input and output paths
    validate_training_paths(config.input_path, config.output_dir)

    # 1. Load and validate data
    df = load_feature_dataset(config.input_path)
    LOGGER.info("Successfully loaded feature dataset: %d rows", len(df))

    # 2. Chronological partition
    split = split_dataset_chronological(df, config.split_config)
    LOGGER.info(
        "Chronological split completed: Train=%d, Val=%d, Test=%d (Disjoint=%s)",
        split.metadata.train_rows,
        split.metadata.val_rows,
        split.metadata.test_rows,
        split.metadata.dates_disjoint,
    )

    # 2b. Validate training class balance (fail early on single-class training set)
    train_classes = np.unique(split.y_train)
    if len(train_classes) < 2:
        raise ModelTrainingError(
            f"Training dataset must contain both binary classes (0 and 1), "
            f"but found only class(es): {train_classes.tolist()}. "
            f"Binary classification models cannot be trained on a single-class dataset."
        )

    # 3. Fit majority-class baseline
    dummy = build_majority_baseline()
    dummy.fit(split.X_train, split.y_train)

    dummy_eval_train = evaluate_model(
        dummy, split.X_train, split.y_train, config.decision_threshold
    )
    dummy_eval_val = evaluate_model(dummy, split.X_val, split.y_val, config.decision_threshold)
    dummy_eval_test = evaluate_model(dummy, split.X_test, split.y_test, config.decision_threshold)

    # 4. Fit Logistic Regression pipeline on Train ONLY
    lr_pipeline = build_logistic_regression_pipeline(
        class_weight=config.class_weight,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    lr_pipeline.fit(split.X_train, split.y_train)

    lr_eval_train = evaluate_model(
        lr_pipeline, split.X_train, split.y_train, config.decision_threshold
    )
    lr_eval_val = evaluate_model(lr_pipeline, split.X_val, split.y_val, config.decision_threshold)
    lr_eval_test = evaluate_model(
        lr_pipeline, split.X_test, split.y_test, config.decision_threshold
    )

    # 5. Compile results
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    delta_acc = round(lr_eval_test["accuracy"] - dummy_eval_test["accuracy"], 6)
    delta_roc = round(lr_eval_test["roc_auc"] - dummy_eval_test["roc_auc"], 6)
    delta_pr = round(lr_eval_test["pr_auc"] - dummy_eval_test["pr_auc"], 6)
    delta_f1 = round(lr_eval_test["f1_score"] - dummy_eval_test["f1_score"], 6)

    metrics: dict[str, Any] = {
        "status": "completed",
        "timestamp": timestamp,
        "target_column": TARGET_COLUMN,
        "delay_threshold_minutes": 15.0,
        "decision_threshold": config.decision_threshold,
        "features": {
            "categorical": list(CATEGORICAL_FEATURES),
            "numerical": list(NUMERICAL_FEATURES),
            "allowlist": list(FEATURE_ALLOWLIST),
            "count": len(FEATURE_ALLOWLIST),
        },
        "model_config": {
            "model_type": "LogisticRegression",
            "solver": "lbfgs",
            "max_iter": config.max_iter,
            "random_state": config.random_state,
            "class_weight": config.class_weight,
        },
        "split_metadata": asdict(split.metadata),
        "models": {
            "majority_baseline": {
                "strategy": "most_frequent",
                "evaluation": {
                    "train": dummy_eval_train,
                    "validation": dummy_eval_val,
                    "test": dummy_eval_test,
                },
            },
            "logistic_regression": {
                "evaluation": {
                    "train": lr_eval_train,
                    "validation": lr_eval_val,
                    "test": lr_eval_test,
                },
            },
        },
        "comparison_summary": {
            "test_accuracy_vs_dummy": delta_acc,
            "test_roc_auc_vs_dummy": delta_roc,
            "test_pr_auc_vs_dummy": delta_pr,
            "test_f1_vs_dummy": delta_f1,
        },
    }

    # 6. Save artifacts
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    dummy_path = output_dir / "dummy_model.joblib"
    metrics_path = output_dir / "evaluation_metrics.json"
    split_path = output_dir / "split_metadata.json"
    report_path = output_dir / "evaluation_report.md"

    joblib.dump(lr_pipeline, model_path)
    joblib.dump(dummy, dummy_path)

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    split_path.write_text(json.dumps(asdict(split.metadata), indent=2) + "\n", encoding="utf-8")

    report_markdown = _generate_markdown_report(metrics)
    report_path.write_text(report_markdown, encoding="utf-8")

    LOGGER.info(
        "Training artifacts successfully written to %s (model, dummy, metrics, split, report)",
        output_dir,
    )

    # 7. MLflow Experiment Tracking
    if config.enable_mlflow:
        mlflow_info = _log_mlflow_run(config, split, lr_pipeline, dummy, metrics, output_dir)
        metrics["mlflow"] = mlflow_info
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    else:
        metrics["mlflow"] = None

    return metrics


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline models for flight delay prediction."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/features/dataset"),
        help="Input feature dataset Parquet path (default: data/features/dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/baseline"),
        help="Output directory for model artifacts (default: models/baseline)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary classification (default: 0.5)",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default=None,
        choices=["balanced", "none"],
        help="Class weight handling for logistic regression (default: none)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum iterations for logistic regression solver (default: 500)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI env var or file:./mlruns)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="flight-delay-baseline",
        help="MLflow experiment name (default: flight-delay-baseline)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (default: auto-generated timestamped name)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for baseline training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _argument_parser().parse_args(argv)
    class_weight = None if args.class_weight in (None, "none") else args.class_weight
    config = TrainingConfig(
        input_path=args.input,
        output_dir=args.output,
        decision_threshold=args.threshold,
        class_weight=class_weight,
        max_iter=args.max_iter,
        random_state=args.random_state,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        enable_mlflow=not args.no_mlflow,
    )
    try:
        results = train_and_evaluate(config)
    except ModelTrainingError as err:
        LOGGER.error("Model training failed: %s", err)
        return 2
    except Exception:
        LOGGER.exception("Unexpected error during model training")
        return 1

    print(json.dumps(results["comparison_summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
