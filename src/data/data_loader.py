"""Data loading and splitting module.

Handles reproducible dataset reading, validation, stratified splitting,
and persistence of processed datasets without modifying raw source files.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_TEST_PATH,
    TARGET_COL,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.data.validation import validate_raw_data
from src.utils.logging_config import get_logger

logger = get_logger("data_loader")


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw dataset from disk and checks schema integrity.

    Args:
        filepath: Path to raw CSV file.

    Returns:
        pd.DataFrame: Loaded raw DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If validation checks fail critically.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")

    logger.info("Loading raw dataset from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded raw dataset with shape %s", df.shape)

    is_valid, issues = validate_raw_data(df)
    if not is_valid:
        logger.warning("Dataset loaded with validation warnings: %s", issues)

    return df


def load_and_split_data(
    filepath: Path = RAW_DATA_PATH,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads raw dataset and splits into stratified train and test partitions.

    Guarantees test partition remains untouched to prevent data leakage.

    Args:
        filepath: Path to raw CSV.
        test_size: Proportion of test partition (default 0.20).
        random_seed: Seed for reproducible split.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    df = load_raw_data(filepath)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    logger.info(
        "Performing stratified train-test split (test_size=%.2f, seed=%d)",
        test_size,
        random_seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    logger.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)
    logger.info(
        "Train target balance: %s (%.1f%% positive)",
        dict(y_train.value_counts()),
        (y_train.mean() * 100),
    )
    logger.info(
        "Test target balance: %s (%.1f%% positive)",
        dict(y_test.value_counts()),
        (y_test.mean() * 100),
    )

    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_path: Path = PROCESSED_TRAIN_PATH,
    test_path: Path = PROCESSED_TEST_PATH,
) -> None:
    """Saves train and test splits to disk."""
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Persisted processed train data to %s", train_path)
    logger.info("Persisted processed test data to %s", test_path)
