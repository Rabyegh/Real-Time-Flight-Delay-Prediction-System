"""Unit tests for Phase 2 feature engineering and leakage prevention."""

from pathlib import Path

import pytest

from flight_delay.processing.feature_engineering import (
    DELAY_THRESHOLD_MINUTES,
    FEATURE_ALLOWLIST,
    PROHIBITED_COLUMNS,
    TARGET_COLUMN,
    FeatureEngineeringError,
    TargetLeakageError,
    assert_no_leakage,
    validate_dataset_paths,
)


def test_feature_allowlist_contains_only_valid_pre_departure_features() -> None:
    assert TARGET_COLUMN not in FEATURE_ALLOWLIST
    assert "DEP_DELAY" not in FEATURE_ALLOWLIST
    assert "DEP_TIME" not in FEATURE_ALLOWLIST
    assert "ARR_DELAY" not in FEATURE_ALLOWLIST
    assert "crs_dep_hour" in FEATURE_ALLOWLIST
    assert "crs_dep_minute" in FEATURE_ALLOWLIST
    assert "carrier" in FEATURE_ALLOWLIST
    assert "origin" in FEATURE_ALLOWLIST
    assert "dest" in FEATURE_ALLOWLIST
    assert "is_weekend" in FEATURE_ALLOWLIST


def test_assert_no_leakage_passes_for_clean_features() -> None:
    # Feature allowlist must pass leakage assertion cleanly
    assert_no_leakage(FEATURE_ALLOWLIST)


@pytest.mark.parametrize("prohibited_col", PROHIBITED_COLUMNS)
def test_assert_no_leakage_rejects_each_prohibited_column(prohibited_col: str) -> None:
    with pytest.raises(TargetLeakageError, match="Prohibited leakage column detected"):
        assert_no_leakage([*FEATURE_ALLOWLIST, prohibited_col])


@pytest.mark.parametrize("prohibited_col", ["dep_delay", "dep_del15", "dep_time", "arr_delay"])
def test_assert_no_leakage_rejects_case_insensitive_prohibited_columns(prohibited_col: str) -> None:
    with pytest.raises(TargetLeakageError):
        assert_no_leakage([*FEATURE_ALLOWLIST, prohibited_col])


def test_assert_no_leakage_rejects_outcome_pattern_variants() -> None:
    with pytest.raises(TargetLeakageError, match="Post-departure actual time field detected"):
        assert_no_leakage(["carrier", "origin", "actual_dep_time_utc"])

    with pytest.raises(TargetLeakageError, match="Delay outcome field detected"):
        assert_no_leakage(["carrier", "origin", "final_arr_delay"])


def test_target_delay_threshold_is_exactly_15_minutes() -> None:
    assert DELAY_THRESHOLD_MINUTES == 15.0


def test_validate_dataset_paths_accepts_valid_separate_paths(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "processed" / "accepted"
    input_path.mkdir(parents=True)
    output_path = tmp_path / "data" / "features"

    in_res, out_res = validate_dataset_paths(input_path, output_path)
    assert in_res == input_path.resolve()
    assert out_res == output_path.resolve()


def test_validate_dataset_paths_rejects_identical_paths(tmp_path: Path) -> None:
    path = tmp_path / "data" / "processed" / "accepted"
    path.mkdir(parents=True)

    with pytest.raises(FeatureEngineeringError, match="Output path cannot be identical"):
        validate_dataset_paths(path, path)


def test_validate_dataset_paths_rejects_output_nested_inside_input(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "processed" / "accepted"
    input_path.mkdir(parents=True)
    nested_output = input_path / "features"

    with pytest.raises(FeatureEngineeringError, match="Output path cannot be inside"):
        validate_dataset_paths(input_path, nested_output)


def test_validate_dataset_paths_rejects_output_parent_of_input(tmp_path: Path) -> None:
    parent_dir = tmp_path / "data"
    input_path = parent_dir / "processed" / "accepted"
    input_path.mkdir(parents=True)

    with pytest.raises(FeatureEngineeringError, match="Output path cannot contain"):
        validate_dataset_paths(input_path, parent_dir)


def test_validate_dataset_paths_rejects_nonexistent_input(tmp_path: Path) -> None:
    input_path = tmp_path / "nonexistent"
    output_path = tmp_path / "features"

    with pytest.raises(FeatureEngineeringError, match="Input path does not exist"):
        validate_dataset_paths(input_path, output_path)
