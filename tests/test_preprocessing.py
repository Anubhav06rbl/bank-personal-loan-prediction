"""Unit tests for feature engineering, cleaning, and preprocessing."""

import pytest
import numpy as np
import pandas as pd
from src.features.feature_engineering import LoanFeatureEngineer, clean_experience_column
from src.features.preprocessing import build_preprocessing_pipeline


def test_clean_experience_column():
    """Tests that negative experience anomalies are properly clamped to 0."""
    sample_df = pd.DataFrame(
        {
            "Age": [24, 25, 30],
            "Experience": [-2, -1, 5],
        }
    )
    cleaned = clean_experience_column(sample_df)
    assert cleaned["Experience"].min() >= 0
    assert list(cleaned["Experience"]) == [0, 0, 5]


def test_loan_feature_engineer_transformation():
    """Tests feature generation logic and dropping of identifier columns."""
    sample_df = pd.DataFrame(
        {
            "ID": [101, 102],
            "Age": [30, 45],
            "Experience": [-1, 20],
            "Income": [50, 120],
            "ZIP Code": [94720, 90089],
            "Family": [2, 4],
            "CCAvg": [2.5, 4.0],
            "Education": [1, 3],
            "Mortgage": [0, 150],
            "Securities Account": [1, 0],
            "CD Account": [0, 1],
            "Online": [1, 1],
            "CreditCard": [0, 1],
        }
    )

    transformer = LoanFeatureEngineer(drop_identifiers=True)
    transformed = transformer.fit_transform(sample_df)

    # Identifiers must be dropped
    assert "ID" not in transformed.columns
    assert "ZIP Code" not in transformed.columns

    # Check Experience cleaned
    assert transformed["Experience"].iloc[0] == 0

    # Check engineered columns exist
    expected_engineered = [
        "Income_Per_Family",
        "CCAvg_to_Income_Ratio",
        "Mortgage_to_Income_Ratio",
        "Has_Mortgage",
        "Total_Accounts",
        "Experience_to_Age_Ratio",
        "High_Networth_Indicator",
    ]
    for col in expected_engineered:
        assert col in transformed.columns, f"Engineered column {col} missing."

    # Validate math on engineered values
    assert transformed["Income_Per_Family"].iloc[0] == pytest.approx(50 / 2, rel=1e-3)
    assert transformed["Has_Mortgage"].tolist() == [0, 1]
    assert transformed["Total_Accounts"].tolist() == [2, 3]  # [1+0+1+0=2, 0+1+1+1=3]


def test_preprocessing_pipeline_fit_transform():
    """Tests that full preprocessing pipeline handles data seamlessly without NaNs."""
    sample_df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "Age": [25, 40, 55],
            "Experience": [-2, 15, 30],
            "Income": [40, 80, 150],
            "ZIP Code": [94720, 90089, 91107],
            "Family": [1, 2, 3],
            "CCAvg": [1.0, 2.0, 5.0],
            "Education": [1, 2, 3],
            "Mortgage": [0, 0, 200],
            "Securities Account": [0, 0, 1],
            "CD Account": [0, 0, 1],
            "Online": [1, 1, 1],
            "CreditCard": [0, 1, 1],
        }
    )

    pipeline = build_preprocessing_pipeline(use_scaler=True)
    transformed = pipeline.fit_transform(sample_df)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.isnull().sum().sum() == 0
    assert "ID" not in transformed.columns
    assert "Income_Per_Family" in transformed.columns
