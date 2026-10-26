"""Inference and prediction module.

Loads the saved best complete pipeline and executes single or batch predictions
with probability scores, decision thresholds, and confidence categories.
"""

from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import json
import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, MODEL_TYPO_PATH, MODEL_METADATA_PATH
from src.data.validation import CustomerLoanInput, BatchCustomerLoanInput
from src.utils.logging_config import get_logger

logger = get_logger("model_predict")


class LoanPredictor:
    """Production inference engine for Bank Personal Loan Acceptance."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.model_path = model_path or MODEL_PATH
        self.metadata_path = metadata_path or MODEL_METADATA_PATH

        # Fallback to typo directory if necessary
        if not self.model_path.exists() and MODEL_TYPO_PATH.exists():
            self.model_path = MODEL_TYPO_PATH

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at {self.model_path}. "
                "Please run `python train_pipeline.py` first."
            )

        logger.info("Loading model artifact from %s", self.model_path)
        self.pipeline = joblib.load(self.model_path)

        # Load metadata and optimal threshold
        self.metadata = self._load_metadata()
        self.decision_threshold = float(self.metadata.get("optimal_threshold", 0.50))
        logger.info(
            "Predictor initialized with model: %s (threshold: %.3f)",
            self.metadata.get("best_model_name", "Unknown"),
            self.decision_threshold,
        )

    def _load_metadata(self) -> Dict[str, Any]:
        """Safely loads model metadata if present."""
        if self.metadata_path and self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read metadata from %s: %s", self.metadata_path, e)
        return {"optimal_threshold": 0.50, "best_model_name": "Pipeline"}

    @staticmethod
    def _categorize_risk(prob: float) -> str:
        """Assigns business propensity segment based on probability score."""
        if prob >= 0.70:
            return "High Propensity"
        elif prob >= 0.35:
            return "Medium Propensity"
        return "Low Propensity"

    def predict_dataframe(self, df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
        """Executes inference on a pandas DataFrame.

        Args:
            df: DataFrame containing customer features.
            threshold: Optional custom decision threshold override.

        Returns:
            pd.DataFrame with added columns: probability, prediction, loan_accepted, propensity_segment
        """
        thresh = threshold if threshold is not None else self.decision_threshold
        probs = self.pipeline.predict_proba(df)[:, 1]
        preds = (probs >= thresh).astype(int)

        result_df = df.copy()
        result_df["probability"] = np.round(probs, 4)
        result_df["prediction"] = preds
        result_df["loan_accepted"] = preds == 1
        result_df["propensity_segment"] = [self._categorize_risk(p) for p in probs]
        return result_df

    def predict_single(self, customer_data: Union[Dict[str, Any], CustomerLoanInput], threshold: Optional[float] = None) -> Dict[str, Any]:
        """Executes prediction for a single customer payload."""
        if isinstance(customer_data, CustomerLoanInput):
            data_dict = customer_data.model_dump(by_alias=True)
        else:
            data_dict = dict(customer_data)

        df = pd.DataFrame([data_dict])
        result = self.predict_dataframe(df, threshold=threshold).iloc[0]

        return {
            "prediction": int(result["prediction"]),
            "loan_accepted": bool(result["loan_accepted"]),
            "acceptance_probability": float(result["probability"]),
            "propensity_segment": str(result["propensity_segment"]),
            "decision_threshold_used": float(threshold if threshold is not None else self.decision_threshold),
        }

    def predict_batch(self, batch_data: Union[List[Dict[str, Any]], BatchCustomerLoanInput], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Executes batch predictions for multiple customers."""
        if isinstance(batch_data, BatchCustomerLoanInput):
            rows = [item.model_dump(by_alias=True) for item in batch_data.customers]
        else:
            rows = list(batch_data)

        df = pd.DataFrame(rows)
        results_df = self.predict_dataframe(df, threshold=threshold)
        return results_df[["probability", "prediction", "loan_accepted", "propensity_segment"]].to_dict(orient="records")
