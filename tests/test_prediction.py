"""Unit and integration tests for model inference and prediction pipeline."""

import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import MODEL_PATH
from src.features.preprocessing import create_full_pipeline
from src.models.predict import LoanPredictor
from src.data.validation import CustomerLoanInput


@pytest.fixture(scope="module")
def sample_customer_data():
    """Provides valid customer input dictionaries for testing."""
    return {
        "Age": 38,
        "Experience": 12,
        "Income": 130,
        "ZIP Code": 94720,
        "Family": 3,
        "CCAvg": 4.5,
        "Education": 3,
        "Mortgage": 0,
        "Securities Account": 0,
        "CD Account": 1,
        "Online": 1,
        "CreditCard": 0,
    }


def test_customer_loan_input_pydantic_validation(sample_customer_data):
    """Tests Pydantic validation on valid and invalid payloads."""
    # Valid instance
    valid_customer = CustomerLoanInput(**sample_customer_data)
    assert valid_customer.Age == 38
    assert valid_customer.Income == 130

    # Invalid instance (negative income)
    invalid_data = sample_customer_data.copy()
    invalid_data["Income"] = -10
    with pytest.raises(Exception):
        CustomerLoanInput(**invalid_data)


def test_loan_predictor_inference(sample_customer_data, tmp_path):
    """Tests LoanPredictor initialization and prediction output format."""
    # Train a minimal pipeline fixture if artifact does not exist yet
    if not MODEL_PATH.exists():
        dummy_df = pd.DataFrame(
            [
                sample_customer_data,
                {**sample_customer_data, "Income": 20, "CCAvg": 0.5, "CD Account": 0},
            ]
        )
        dummy_y = pd.Series([1, 0])
        pipe = create_full_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
        pipe.fit(dummy_df, dummy_y)
        temp_model_file = tmp_path / "temp_model.joblib"
        import joblib
        joblib.dump(pipe, temp_model_file)
        predictor = LoanPredictor(model_path=temp_model_file)
    else:
        predictor = LoanPredictor()

    result = predictor.predict_single(sample_customer_data)

    assert "prediction" in result
    assert result["prediction"] in (0, 1)
    assert "acceptance_probability" in result
    assert 0.0 <= result["acceptance_probability"] <= 1.0
    assert "propensity_segment" in result
    assert result["propensity_segment"] in ("High Propensity", "Medium Propensity", "Low Propensity")
    assert "loan_accepted" in result
    assert isinstance(result["loan_accepted"], bool)


def test_loan_predictor_batch_inference(sample_customer_data, tmp_path):
    """Tests batch prediction capabilities."""
    if not MODEL_PATH.exists():
        dummy_df = pd.DataFrame([sample_customer_data, sample_customer_data])
        dummy_y = pd.Series([1, 0])
        pipe = create_full_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
        pipe.fit(dummy_df, dummy_y)
        temp_model_file = tmp_path / "temp_model.joblib"
        import joblib
        joblib.dump(pipe, temp_model_file)
        predictor = LoanPredictor(model_path=temp_model_file)
    else:
        predictor = LoanPredictor()

    batch = [sample_customer_data, sample_customer_data]
    results = predictor.predict_batch(batch)

    assert len(results) == 2
    assert "probability" in results[0]
    assert "prediction" in results[0]
