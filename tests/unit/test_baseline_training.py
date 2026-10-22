"""Unit tests for Phase 3 baseline model training, evaluation, and data split."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flight_delay.processing.feature_engineering import FEATURE_ALLOWLIST, TARGET_COLUMN
from flight_delay.training.baseline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    DataSplitError,
    MLflowTrackingError,
    ModelTrainingError,
    PathSafetyError,
    SchemaValidationError,
    SplitConfig,
    assert_feature_columns,
    build_logistic_regression_pipeline,
    build_majority_baseline,
    build_preprocessing_pipeline,
    evaluate_predictions,
    load_feature_dataset,
    split_dataset_chronological,
    train_and_evaluate,
    validate_experiment_name,
    validate_split_config,
    validate_tracking_uri,
    validate_training_paths,
)


def _make_sample_row(
    month: int = 2,
    day_of_month: int = 15,
    is_delayed_15: int = 0,
    carrier: str = "AA",
    origin: str = "JFK",
    dest: str = "BOS",
) -> dict:
    return {
        "is_delayed_15": is_delayed_15,
        "carrier": carrier,
        "origin": origin,
        "dest": dest,
        "crs_dep_hour": 8,
        "crs_dep_minute": 30,
        "dep_time_blk": "0800-0859",
        "crs_arr_hour": 10,
        "crs_arr_minute": 0,
        "arr_time_blk": "1000-1059",
        "month": month,
        "day_of_month": day_of_month,
        "day_of_week": 3,
        "is_weekend": 0,
        "crs_elapsed_time": 90.0,
        "distance": 200.0,
    }


def test_feature_allowlist_matches_categorical_and_numerical_sets() -> None:
    combined = set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES)
    assert combined == set(FEATURE_ALLOWLIST)
    assert len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES) == len(FEATURE_ALLOWLIST)
    assert TARGET_COLUMN not in combined


def test_assert_feature_columns_passes_for_valid_features() -> None:
    assert_feature_columns(FEATURE_ALLOWLIST)


def test_assert_feature_columns_rejects_missing_column() -> None:
    incomplete = [c for c in FEATURE_ALLOWLIST if c != "carrier"]
    with pytest.raises(SchemaValidationError, match="Missing required feature columns"):
        assert_feature_columns(incomplete)


def test_assert_feature_columns_rejects_unexpected_columns() -> None:
    with pytest.raises(SchemaValidationError, match="Unexpected unapproved columns"):
        assert_feature_columns([*FEATURE_ALLOWLIST, "unapproved_feature"])


def test_assert_feature_columns_permits_target_when_flagged() -> None:
    assert_feature_columns([TARGET_COLUMN, *FEATURE_ALLOWLIST], allow_target=True)
    with pytest.raises(SchemaValidationError, match="Unexpected unapproved columns"):
        assert_feature_columns([TARGET_COLUMN, *FEATURE_ALLOWLIST], allow_target=False)


@pytest.mark.parametrize("prohibited", ["DEP_DELAY", "dep_time", "ARR_DELAY", "FL_DATE"])
def test_assert_feature_columns_rejects_prohibited_or_leakage_columns(prohibited: str) -> None:
    with pytest.raises(SchemaValidationError):
        assert_feature_columns([*FEATURE_ALLOWLIST, prohibited])


def test_load_feature_dataset_rejects_nonexistent_path(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match="does not exist"):
        load_feature_dataset(tmp_path / "nonexistent")


def test_load_feature_dataset_rejects_empty_dataset(tmp_path: Path) -> None:
    empty_df = pd.DataFrame(columns=[TARGET_COLUMN, *FEATURE_ALLOWLIST])
    parquet_path = tmp_path / "empty.parquet"
    empty_df.to_parquet(parquet_path)
    with pytest.raises(SchemaValidationError, match="empty"):
        load_feature_dataset(parquet_path)


def test_load_feature_dataset_rejects_non_binary_target(tmp_path: Path) -> None:
    rows = [_make_sample_row(is_delayed_15=0), _make_sample_row(is_delayed_15=2)]
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "invalid_target.parquet"
    df.to_parquet(parquet_path)
    with pytest.raises(SchemaValidationError, match="must contain only binary values"):
        load_feature_dataset(parquet_path)


def test_load_feature_dataset_rejects_nulls_in_features(tmp_path: Path) -> None:
    row = _make_sample_row()
    row["distance"] = None
    df = pd.DataFrame([row])
    parquet_path = tmp_path / "null_feature.parquet"
    df.to_parquet(parquet_path)
    with pytest.raises(SchemaValidationError, match="Feature columns contain null values"):
        load_feature_dataset(parquet_path)


def test_split_dataset_chronological_ensures_date_disjointness_and_counts() -> None:
    records = [
        # Train: Feb 1-28 + June 1-16
        _make_sample_row(month=2, day_of_month=1, is_delayed_15=0),
        _make_sample_row(month=2, day_of_month=15, is_delayed_15=1),
        _make_sample_row(month=6, day_of_month=5, is_delayed_15=0),
        _make_sample_row(month=6, day_of_month=16, is_delayed_15=1),
        # Val: June 17-23
        _make_sample_row(month=6, day_of_month=17, is_delayed_15=0),
        _make_sample_row(month=6, day_of_month=20, is_delayed_15=1),
        # Test: June 24-30
        _make_sample_row(month=6, day_of_month=25, is_delayed_15=0),
        _make_sample_row(month=6, day_of_month=30, is_delayed_15=1),
    ]
    df = pd.DataFrame(records)
    split = split_dataset_chronological(df, config=SplitConfig())

    assert split.metadata.train_rows == 4
    assert split.metadata.val_rows == 2
    assert split.metadata.test_rows == 2
    assert split.metadata.dates_disjoint is True

    # Confirm strict date separation
    train_dates = set(zip(split.X_train["month"], split.X_train["day_of_month"], strict=False))
    val_dates = set(zip(split.X_val["month"], split.X_val["day_of_month"], strict=False))
    test_dates = set(zip(split.X_test["month"], split.X_test["day_of_month"], strict=False))

    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)


def test_split_dataset_chronological_rejects_empty_split() -> None:
    # Dataset only contains February; June test dates are missing
    records = [
        _make_sample_row(month=2, day_of_month=1),
        _make_sample_row(month=2, day_of_month=15),
    ]
    df = pd.DataFrame(records)
    with pytest.raises(DataSplitError, match="Validation split is empty"):
        split_dataset_chronological(df)


def test_preprocessing_pipeline_handles_unseen_categories() -> None:
    train_rows = [
        _make_sample_row(carrier="AA", origin="JFK", dest="BOS"),
        _make_sample_row(carrier="DL", origin="ATL", dest="MIA"),
    ]
    test_rows = [
        # Unseen carrier, origin, and destination
        _make_sample_row(carrier="UA", origin="SFO", dest="HNL"),
    ]

    X_train = pd.DataFrame(train_rows)[list(FEATURE_ALLOWLIST)]
    X_test = pd.DataFrame(test_rows)[list(FEATURE_ALLOWLIST)]

    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(X_train)
    transformed = preprocessor.transform(X_test)

    # Must transform without error into sparse matrix
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > len(NUMERICAL_FEATURES)


def test_evaluate_predictions_computes_all_metrics() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    y_prob = [0.1, 0.6, 0.4, 0.9]

    metrics = evaluate_predictions(y_true, y_pred, y_prob=y_prob, threshold=0.5)

    assert metrics["total_samples"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 2
    assert metrics["positive_prevalence"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["confusion_matrix"]["true_negative"] == 1
    assert metrics["confusion_matrix"]["false_positive"] == 1
    assert metrics["confusion_matrix"]["false_negative"] == 1
    assert metrics["confusion_matrix"]["true_positive"] == 1


def test_evaluate_predictions_handles_single_class_edge_case() -> None:
    # Edge case: All samples are negative (class 0 only)
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    y_prob = [0.1, 0.2, 0.15, 0.05]

    metrics = evaluate_predictions(y_true, y_pred, y_prob=y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 0.5  # Graceful fallback when single class
    assert metrics["positive_prevalence"] == 0.0
    assert metrics["positive_count"] == 0


def test_majority_baseline_predicts_most_frequent_class() -> None:
    # 75% negative, 25% positive
    train_rows = [
        _make_sample_row(is_delayed_15=0),
        _make_sample_row(is_delayed_15=0),
        _make_sample_row(is_delayed_15=0),
        _make_sample_row(is_delayed_15=1),
    ]
    df = pd.DataFrame(train_rows)
    X = df[list(FEATURE_ALLOWLIST)]
    y = df[TARGET_COLUMN]

    dummy = build_majority_baseline()
    dummy.fit(X, y)
    preds = dummy.predict(X)

    # Must predict 0 for all rows
    assert np.all(preds == 0)


def test_reproducibility_of_training_with_fixed_seed() -> None:
    # Generating 20 rows of synthetic data
    np.random.seed(42)
    rows = []
    for i in range(30):
        rows.append(
            _make_sample_row(
                month=2 if i < 20 else 6,
                day_of_month=1 + (i % 25),
                is_delayed_15=i % 3 == 0,
                carrier="AA" if i % 2 == 0 else "DL",
            )
        )
    df = pd.DataFrame(rows)
    X = df[list(FEATURE_ALLOWLIST)]
    y = df[TARGET_COLUMN].astype(int)

    pipeline_a = build_logistic_regression_pipeline(random_state=42, max_iter=100)
    pipeline_a.fit(X, y)

    pipeline_b = build_logistic_regression_pipeline(random_state=42, max_iter=100)
    pipeline_b.fit(X, y)

    coef_a = pipeline_a.named_steps["classifier"].coef_
    coef_b = pipeline_b.named_steps["classifier"].coef_

    np.testing.assert_allclose(coef_a, coef_b, rtol=1e-5)


def test_validate_training_paths_safety(tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Valid disjoint paths
    in_res, out_res = validate_training_paths(data_dir, models_dir)
    assert in_res == data_dir.resolve()
    assert out_res == models_dir.resolve()

    # Identical path
    with pytest.raises(PathSafetyError, match="cannot be identical"):
        validate_training_paths(data_dir, data_dir)

    # Output nested inside input
    nested_out = data_dir / "nested_models"
    with pytest.raises(PathSafetyError, match="cannot be nested inside"):
        validate_training_paths(data_dir, nested_out)

    # Output contains input
    with pytest.raises(PathSafetyError, match="cannot contain"):
        validate_training_paths(data_dir, tmp_path)

    # Missing input
    with pytest.raises(SchemaValidationError, match="does not exist"):
        validate_training_paths(tmp_path / "nonexistent", models_dir)


def test_validate_split_config_rejects_invalid_and_overlapping_configs() -> None:
    # Day out of range
    with pytest.raises(DataSplitError, match="Invalid train_june_end_day"):
        validate_split_config(SplitConfig(train_june_end_day=32))

    with pytest.raises(DataSplitError, match="Invalid validation day range"):
        validate_split_config(SplitConfig(val_start_day=25, val_end_day=20))

    with pytest.raises(DataSplitError, match="Invalid test day range"):
        validate_split_config(SplitConfig(test_start_day=0, test_end_day=10))

    # Overlap between val and test in same month
    with pytest.raises(DataSplitError, match="boundaries overlap or are inverted"):
        validate_split_config(SplitConfig(val_start_day=17, val_end_day=25, test_start_day=24))

    # Overlap between train and val
    with pytest.raises(DataSplitError, match="Train and validation boundaries overlap"):
        validate_split_config(SplitConfig(train_june_end_day=18, val_start_day=17))


def test_split_dataset_chronological_rejects_unassigned_rows_and_unsupported_months() -> None:
    # Month 3 is unsupported in default config
    records = [
        _make_sample_row(month=2, day_of_month=1),
        _make_sample_row(month=3, day_of_month=15),
        _make_sample_row(month=6, day_of_month=20),
        _make_sample_row(month=6, day_of_month=28),
    ]
    df = pd.DataFrame(records)
    with pytest.raises(
        DataSplitError, match="fall outside configured chronological split boundaries"
    ):
        split_dataset_chronological(df)


def test_split_dataset_chronological_respects_custom_train_months() -> None:
    records = [
        _make_sample_row(month=1, day_of_month=10),
        _make_sample_row(month=2, day_of_month=10),
        _make_sample_row(month=6, day_of_month=10),
        _make_sample_row(month=6, day_of_month=20),
        _make_sample_row(month=6, day_of_month=28),
    ]
    df = pd.DataFrame(records)
    custom_config = SplitConfig(train_months=(1, 2, 6))
    split = split_dataset_chronological(df, config=custom_config)

    assert split.metadata.train_rows == 3
    assert set(split.X_train["month"].unique()) == {1, 2, 6}
    assert split.metadata.train_dates.startswith("2023-01-01")


def test_train_and_evaluate_rejects_single_class_training_set(tmp_path: Path) -> None:
    records = [
        # Train (all 0)
        _make_sample_row(month=2, day_of_month=1, is_delayed_15=0),
        _make_sample_row(month=2, day_of_month=15, is_delayed_15=0),
        _make_sample_row(month=6, day_of_month=5, is_delayed_15=0),
        _make_sample_row(month=6, day_of_month=16, is_delayed_15=0),
        # Val (has 1)
        _make_sample_row(month=6, day_of_month=17, is_delayed_15=0),
        _make_sample_row(month=6, day_of_month=20, is_delayed_15=1),
        # Test (has 1)
        _make_sample_row(month=6, day_of_month=25, is_delayed_15=0),
        _make_sample_row(month=6, day_of_month=30, is_delayed_15=1),
    ]
    df = pd.DataFrame(records)
    dataset_path = tmp_path / "single_class_train.parquet"
    df.to_parquet(dataset_path)

    from flight_delay.training.baseline import TrainingConfig

    config = TrainingConfig(
        input_path=dataset_path,
        output_dir=tmp_path / "models",
    )
    with pytest.raises(
        ModelTrainingError, match="Training dataset must contain both binary classes"
    ):
        train_and_evaluate(config)


def test_validate_tracking_uri_accepts_valid_schemes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert validate_tracking_uri("file:./mlruns") == "file:./mlruns"
    assert validate_tracking_uri("sqlite:///mlruns.db") == "sqlite:///mlruns.db"
    assert validate_tracking_uri("http://localhost:5000") == "http://localhost:5000"
    assert validate_tracking_uri("https://mlflow.example.com") == "https://mlflow.example.com"
    assert validate_tracking_uri("mlruns") == "mlruns"

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert validate_tracking_uri(None) == "file:./mlruns"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///env_store.db")
    assert validate_tracking_uri(None) == "sqlite:///env_store.db"


def test_validate_tracking_uri_rejects_empty_and_unsupported_schemes() -> None:
    with pytest.raises(MLflowTrackingError, match="cannot be empty or whitespace"):
        validate_tracking_uri("")

    with pytest.raises(MLflowTrackingError, match="cannot be empty or whitespace"):
        validate_tracking_uri("   ")

    with pytest.raises(
        MLflowTrackingError, match="Unsupported or invalid MLflow tracking URI scheme"
    ):
        validate_tracking_uri("ftp://invalid-server/mlruns")


def test_validate_experiment_name_accepts_valid_and_rejects_empty() -> None:
    assert validate_experiment_name("my-experiment") == "my-experiment"
    assert validate_experiment_name(None) == "flight-delay-baseline"

    with pytest.raises(MLflowTrackingError, match="cannot be empty or whitespace"):
        validate_experiment_name("")

    with pytest.raises(MLflowTrackingError, match="cannot be empty or whitespace"):
        validate_experiment_name("   ")
