"""Integration tests for baseline training pipeline, CLI, and artifact persistence."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import pytest

from flight_delay.processing.feature_engineering import FEATURE_ALLOWLIST
from flight_delay.training.baseline import (
    PathSafetyError,
    TrainingConfig,
    main,
    train_and_evaluate,
)


def _generate_synthetic_feature_dataset(path: Path) -> Path:
    """Generate a valid, multi-date synthetic feature dataset in Parquet format."""
    records = []
    # Train dates: Feb 1..20, June 1..16
    for day in range(1, 21):
        for carrier in ("AA", "DL", "UA"):
            records.append(
                {
                    "is_delayed_15": 1 if day % 4 == 0 else 0,
                    "carrier": carrier,
                    "origin": "JFK" if carrier == "AA" else "ATL",
                    "dest": "BOS" if carrier == "AA" else "MIA",
                    "crs_dep_hour": (day * 2) % 24,
                    "crs_dep_minute": 15,
                    "dep_time_blk": "0800-0859",
                    "crs_arr_hour": (day * 2 + 2) % 24,
                    "crs_arr_minute": 45,
                    "arr_time_blk": "1000-1059",
                    "month": 2,
                    "day_of_month": day,
                    "day_of_week": (day % 7) + 1,
                    "is_weekend": 1 if (day % 7) in (5, 6) else 0,
                    "crs_elapsed_time": 90.0,
                    "distance": 250.0,
                }
            )
    for day in range(1, 17):
        for carrier in ("AA", "DL", "UA"):
            records.append(
                {
                    "is_delayed_15": 1 if day % 3 == 0 else 0,
                    "carrier": carrier,
                    "origin": "ORD",
                    "dest": "DFW",
                    "crs_dep_hour": 12,
                    "crs_dep_minute": 0,
                    "dep_time_blk": "1200-1259",
                    "crs_arr_hour": 14,
                    "crs_arr_minute": 30,
                    "arr_time_blk": "1400-1459",
                    "month": 6,
                    "day_of_month": day,
                    "day_of_week": (day % 7) + 1,
                    "is_weekend": 1 if (day % 7) in (5, 6) else 0,
                    "crs_elapsed_time": 150.0,
                    "distance": 800.0,
                }
            )
    # Val dates: June 17..23
    for day in range(17, 24):
        for carrier in ("AA", "DL", "UA"):
            records.append(
                {
                    "is_delayed_15": 1 if day % 2 == 0 else 0,
                    "carrier": carrier,
                    "origin": "JFK",
                    "dest": "LAX",
                    "crs_dep_hour": 9,
                    "crs_dep_minute": 0,
                    "dep_time_blk": "0900-0959",
                    "crs_arr_hour": 12,
                    "crs_arr_minute": 30,
                    "arr_time_blk": "1200-1259",
                    "month": 6,
                    "day_of_month": day,
                    "day_of_week": (day % 7) + 1,
                    "is_weekend": 1 if (day % 7) in (5, 6) else 0,
                    "crs_elapsed_time": 330.0,
                    "distance": 2400.0,
                }
            )
    # Test dates: June 24..30
    for day in range(24, 31):
        for carrier in ("AA", "DL", "UA"):
            records.append(
                {
                    "is_delayed_15": 1 if day % 2 == 0 else 0,
                    "carrier": carrier,
                    "origin": "BOS",
                    "dest": "SFO",
                    "crs_dep_hour": 15,
                    "crs_dep_minute": 45,
                    "dep_time_blk": "1500-1559",
                    "crs_arr_hour": 19,
                    "crs_arr_minute": 15,
                    "arr_time_blk": "1900-1959",
                    "month": 6,
                    "day_of_month": day,
                    "day_of_week": (day % 7) + 1,
                    "is_weekend": 1 if (day % 7) in (5, 6) else 0,
                    "crs_elapsed_time": 390.0,
                    "distance": 2700.0,
                }
            )

    df = pd.DataFrame(records)
    path.mkdir(parents=True, exist_ok=True)
    dataset_file = path / "data.parquet"
    df.to_parquet(dataset_file)
    return dataset_file


def test_full_training_pipeline_generates_all_artifacts_and_metrics(tmp_path: Path) -> None:
    dataset_file = _generate_synthetic_feature_dataset(tmp_path / "features")
    output_dir = tmp_path / "models" / "baseline"

    config = TrainingConfig(
        input_path=dataset_file,
        output_dir=output_dir,
        max_iter=100,
        random_state=42,
        tracking_uri=(tmp_path / "mlruns").as_uri(),
    )

    metrics = train_and_evaluate(config)

    assert metrics["status"] == "completed"
    assert "models" in metrics
    assert "logistic_regression" in metrics["models"]
    assert "majority_baseline" in metrics["models"]

    # Verify all artifacts on disk
    model_file = output_dir / "model.joblib"
    dummy_file = output_dir / "dummy_model.joblib"
    metrics_file = output_dir / "evaluation_metrics.json"
    split_file = output_dir / "split_metadata.json"
    report_file = output_dir / "evaluation_report.md"

    assert model_file.is_file()
    assert dummy_file.is_file()
    assert metrics_file.is_file()
    assert split_file.is_file()
    assert report_file.is_file()

    # Verify model is loadable and can predict
    pipeline = joblib.load(model_file)
    test_sample = pd.DataFrame(
        [
            {
                "carrier": "AA",
                "origin": "JFK",
                "dest": "BOS",
                "crs_dep_hour": 8,
                "crs_dep_minute": 30,
                "dep_time_blk": "0800-0859",
                "crs_arr_hour": 10,
                "crs_arr_minute": 0,
                "arr_time_blk": "1000-1059",
                "month": 6,
                "day_of_month": 28,
                "day_of_week": 4,
                "is_weekend": 0,
                "crs_elapsed_time": 90.0,
                "distance": 200.0,
            }
        ]
    )[list(FEATURE_ALLOWLIST)]

    prediction = pipeline.predict(test_sample)
    proba = pipeline.predict_proba(test_sample)
    assert prediction[0] in (0, 1)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0

    # Verify report contents
    report_text = report_file.read_text(encoding="utf-8")
    assert "# Baseline Model Evaluation Report" in report_text
    assert "Accuracy" in report_text
    assert "ROC-AUC" in report_text
    assert "PR-AUC" in report_text
    assert "Limitations & Non-Production Notice" in report_text
    assert "Validation Set Evaluation Comparison" in report_text
    assert "Role of Validation Partition" in report_text


def test_full_training_pipeline_rejects_colliding_paths(tmp_path: Path) -> None:
    dataset_file = _generate_synthetic_feature_dataset(tmp_path / "features")
    feature_dir = dataset_file.parent

    # Try setting output_dir equal to feature dataset directory
    config = TrainingConfig(
        input_path=feature_dir,
        output_dir=feature_dir,
    )
    with pytest.raises(PathSafetyError, match="cannot be identical"):
        train_and_evaluate(config)

    # Try setting output_dir inside feature dataset directory
    nested_config = TrainingConfig(
        input_path=feature_dir,
        output_dir=feature_dir / "models_nested",
    )
    with pytest.raises(PathSafetyError, match="cannot be nested inside"):
        train_and_evaluate(nested_config)

    # Confirm original parquet file was not modified or corrupted
    assert dataset_file.is_file()


def test_cli_execution_with_arguments(tmp_path: Path) -> None:
    dataset_file = _generate_synthetic_feature_dataset(tmp_path / "features")
    output_dir = tmp_path / "models" / "cli_output"

    exit_code = main(
        [
            "--input",
            str(dataset_file),
            "--output",
            str(output_dir),
            "--max-iter",
            "50",
            "--random-state",
            "42",
            "--tracking-uri",
            (tmp_path / "mlruns").as_uri(),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "evaluation_metrics.json").exists()


def test_cli_fails_on_colliding_paths(tmp_path: Path) -> None:
    dataset_file = _generate_synthetic_feature_dataset(tmp_path / "features")
    exit_code = main(
        [
            "--input",
            str(dataset_file.parent),
            "--output",
            str(dataset_file.parent),
        ]
    )
    assert exit_code == 2


def test_cli_fails_on_missing_dataset(tmp_path: Path) -> None:
    nonexistent = tmp_path / "missing_data"
    exit_code = main(["--input", str(nonexistent)])
    assert exit_code == 2


def test_full_training_pipeline_logs_to_mlflow_local_store(tmp_path: Path) -> None:
    dataset_file = _generate_synthetic_feature_dataset(tmp_path / "features")
    output_dir = tmp_path / "models" / "baseline"
    mlruns_dir = tmp_path / "mlruns"
    tracking_uri = mlruns_dir.as_uri()

    config = TrainingConfig(
        input_path=dataset_file,
        output_dir=output_dir,
        max_iter=50,
        random_state=42,
        tracking_uri=tracking_uri,
        experiment_name="test-flight-delay",
        run_name="test-run-baseline",
        enable_mlflow=True,
    )

    metrics = train_and_evaluate(config)
    assert metrics["mlflow"] is not None
    run_id = metrics["mlflow"]["run_id"]
    experiment_id = metrics["mlflow"]["experiment_id"]

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert run.info.experiment_id == experiment_id
    assert run.data.tags["model_family"] == "baseline"
    assert run.data.tags["algorithm"] == "logistic_regression"

    # Verify parameters
    assert run.data.params["model_type"] == "LogisticRegression"
    assert run.data.params["max_iter"] == "50"
    assert run.data.params["feature_count"] == "15"
    assert run.data.params["target_column"] == "is_delayed_15"

    # Verify separated metrics
    for metric_name in (
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "val_roc_auc",
        "test_roc_auc",
        "val_pr_auc",
        "test_pr_auc",
        "test_accuracy_vs_dummy",
    ):
        assert metric_name in run.data.metrics
        assert isinstance(run.data.metrics[metric_name], float)

    # Verify artifacts
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    assert "model" in artifacts
    assert "artifacts" in artifacts

    sub_artifacts = [a.path for a in client.list_artifacts(run_id, path="artifacts")]
    for expected in (
        "artifacts/model.joblib",
        "artifacts/dummy_model.joblib",
        "artifacts/evaluation_metrics.json",
        "artifacts/split_metadata.json",
        "artifacts/evaluation_report.md",
    ):
        assert expected in sub_artifacts

    # Verify model artifact can be reloaded via mlflow.sklearn
    loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    test_sample = pd.DataFrame(
        [
            {
                "carrier": "AA",
                "origin": "JFK",
                "dest": "BOS",
                "crs_dep_hour": 8,
                "crs_dep_minute": 30,
                "dep_time_blk": "0800-0859",
                "crs_arr_hour": 10,
                "crs_arr_minute": 0,
                "arr_time_blk": "1000-1059",
                "month": 6,
                "day_of_month": 28,
                "day_of_week": 4,
                "is_weekend": 0,
                "crs_elapsed_time": 90.0,
                "distance": 200.0,
            }
        ]
    )[list(FEATURE_ALLOWLIST)]
    prediction = loaded_model.predict(test_sample)
    assert prediction[0] in (0, 1)


def test_training_pipeline_with_mlflow_disabled(tmp_path: Path) -> None:
    dataset_file = _generate_synthetic_feature_dataset(tmp_path / "features")
    output_dir = tmp_path / "models" / "baseline"

    config = TrainingConfig(
        input_path=dataset_file,
        output_dir=output_dir,
        max_iter=50,
        random_state=42,
        enable_mlflow=False,
    )

    metrics = train_and_evaluate(config)
    assert metrics["status"] == "completed"
    assert metrics["mlflow"] is None


def test_cli_execution_with_mlflow_flags(tmp_path: Path) -> None:
    dataset_file = _generate_synthetic_feature_dataset(tmp_path / "features")
    output_dir = tmp_path / "models" / "cli_output"
    mlruns_dir = tmp_path / "mlruns"
    tracking_uri = mlruns_dir.as_uri()

    exit_code = main(
        [
            "--input",
            str(dataset_file),
            "--output",
            str(output_dir),
            "--max-iter",
            "50",
            "--random-state",
            "42",
            "--tracking-uri",
            tracking_uri,
            "--experiment-name",
            "cli-experiment",
            "--run-name",
            "cli-run-1",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "model.joblib").exists()
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("cli-experiment")
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"


def test_cli_fails_on_invalid_tracking_uri(tmp_path: Path) -> None:
    dataset_file = _generate_synthetic_feature_dataset(tmp_path / "features")
    output_dir = tmp_path / "models" / "cli_output"

    exit_code = main(
        [
            "--input",
            str(dataset_file),
            "--output",
            str(output_dir),
            "--tracking-uri",
            "ftp://invalid-server/tracking",
        ]
    )
    assert exit_code == 2
