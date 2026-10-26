"""FastAPI inference service for Bank Personal Loan Acceptance Prediction.

Provides RESTful endpoints for healthchecks, model metadata, single customer
prediction, and high-throughput batch evaluation.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.validation import CustomerLoanInput, BatchCustomerLoanInput
from src.models.predict import LoanPredictor
from src.utils.logging_config import get_logger

logger = get_logger("fastapi_app")

app = FastAPI(
    title="Bank Personal Loan Prediction API",
    description=(
        "Production ML Inference Service predicting customer acceptance probability "
        "for Personal Loan offers using tree-based ensemble models."
    ),
    version="1.0.0",
)

# Enable CORS for local testing & web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[LoanPredictor] = None


@app.on_event("startup")
def load_model_on_startup():
    """Initializes the model predictor upon API startup."""
    global predictor
    try:
        predictor = LoanPredictor()
        logger.info("LoanPredictor successfully initialized on API startup.")
    except Exception as e:
        logger.error("Failed to initialize LoanPredictor on startup: %s", e)
        predictor = None


@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    """Root endpoint welcoming users and providing documentation link."""
    return {
        "service": "Bank Personal Loan Prediction API",
        "status": "online",
        "documentation": "/docs",
    }


@app.get("/health", tags=["Health"])
def health() -> Dict[str, Any]:
    """Healthcheck endpoint verifying model readiness."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please execute `python train_pipeline.py` first.",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "champion_model": predictor.metadata.get("best_model_name", "Unknown"),
        "decision_threshold": predictor.decision_threshold,
    }


@app.get("/metadata", tags=["Model Information"])
def get_metadata() -> Dict[str, Any]:
    """Retrieves metadata, training metrics, and configuration for the active model."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return predictor.metadata


class PredictionResponse(BaseModel):
    """Output schema for a single prediction request."""

    prediction: int = Field(..., description="0 = No Loan, 1 = Personal Loan Accepted")
    loan_accepted: bool = Field(..., description="True if customer accepted loan")
    acceptance_probability: float = Field(..., description="Confidence score [0.0 - 1.0]")
    propensity_segment: str = Field(..., description="High, Medium, or Low Propensity")
    decision_threshold_used: float = Field(..., description="Decision boundary used")


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(customer: CustomerLoanInput, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Predicts loan acceptance likelihood for an individual customer.

    - **customer**: Customer demographics, financials, and bank relationship info.
    - **threshold**: Optional probability cutoff override (default uses optimal calibrated threshold).
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded.",
        )
    try:
        return predictor.predict_single(customer, threshold=threshold)
    except Exception as e:
        logger.error("Inference error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict_batch", tags=["Inference"])
def predict_batch(payload: BatchCustomerLoanInput, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Batch prediction endpoint for marketing campaign targeting."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded.",
        )
    try:
        return predictor.predict_batch(payload, threshold=threshold)
    except Exception as e:
        logger.error("Batch inference error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
