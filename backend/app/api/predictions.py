"""Prediction API routes."""

from fastapi import APIRouter

from backend.app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(
    prefix="/predict",
    tags=["prediction"],
)


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict software defect risk",
)
def predict_defect_risk(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Validate software metrics for defect-risk prediction.

    Trained-model inference will be connected in a later integration step.
    """

    return PredictionResponse(
        status="pending_model_integration",
        defect_probability=None,
        risk_level="NOT_AVAILABLE",
        model_version=None,
    )
