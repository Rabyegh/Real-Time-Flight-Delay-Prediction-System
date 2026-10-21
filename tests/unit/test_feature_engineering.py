"""Unit tests for Phase 2 feature engineering and leakage prevention."""

import pytest

from flight_delay.processing.feature_engineering import (
    DELAY_THRESHOLD_MINUTES,
    FEATURE_ALLOWLIST,
    PROHIBITED_COLUMNS,
    TARGET_COLUMN,
    TargetLeakageError,
    assert_no_leakage,
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
