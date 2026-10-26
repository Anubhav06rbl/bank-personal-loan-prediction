"""Unit tests for data ingestion, loading, and validation."""

import pytest
import pandas as pd
from src.config import RAW_DATA_PATH, TARGET_COL, FEATURE_COLS
from src.data.data_loader import load_raw_data, load_and_split_data
from src.data.validation import validate_raw_data


def test_raw_data_file_exists():
    """Verifies that the raw data file exists at the expected path."""
    assert RAW_DATA_PATH.exists(), f"Raw data file not found at {RAW_DATA_PATH}"


def test_load_raw_data():
    """Tests loading raw data and verifying basic shape and columns."""
    df = load_raw_data(RAW_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 5000
    assert df.shape[1] == 14
    assert TARGET_COL in df.columns
    assert "ID" in df.columns


def test_validate_raw_data():
    """Tests the validation function on clean and corrupted data."""
    df = load_raw_data(RAW_DATA_PATH)
    is_valid, issues = validate_raw_data(df)
    assert is_valid is True
    assert len(issues) == 0

    # Corrupt a column and check validation failure
    corrupt_df = df.drop(columns=[TARGET_COL])
    is_valid_c, issues_c = validate_raw_data(corrupt_df)
    assert is_valid_c is False
    assert any("Missing required columns" in issue for issue in issues_c)


def test_load_and_split_data_reproducibility():
    """Tests stratified split proportions and exact seed reproducibility."""
    X_train1, X_test1, y_train1, y_test1 = load_and_split_data(RAW_DATA_PATH, test_size=0.2, random_seed=42)
    X_train2, X_test2, y_train2, y_test2 = load_and_split_data(RAW_DATA_PATH, test_size=0.2, random_seed=42)

    # Check shapes
    assert len(X_train1) == 4000
    assert len(X_test1) == 1000
    assert len(y_train1) == 4000
    assert len(y_test1) == 1000

    # Check identical split for seed 42
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)

    # Check stratification: target proportion should be approximately equal
    train_prop = y_train1.mean()
    test_prop = y_test1.mean()
    assert abs(train_prop - test_prop) < 0.01
    assert round(train_prop, 2) == 0.10
