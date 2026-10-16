"""Prediction API routes."""

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.prediction import (
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.services.model_service import (
    FeatureValidationError,
    ModelArtifactError,
    ModelNotReadyError,
    model_service,
)

router = APIRouter(tags=["prediction"])


@router.get("/model/info", response_model=ModelInfoResponse)
@router.get("/model-info", response_model=ModelInfoResponse, include_in_schema=False)
def model_info() -> ModelInfoResponse:
    """Return model status, expected features, thresholds, and evaluation metadata."""
    return ModelInfoResponse(**model_service.get_model_info())


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict software defect risk",
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Prediction metrics failed validation."},
        503: {"description": "The configured model artifact is unavailable or invalid."},
    },
)
def predict_defect_risk(request: PredictionRequest) -> PredictionResponse:
    """Validate metrics and run defect-risk inference with the configured model."""
    try:
        result = model_service.predict(request.model_dump())
        return PredictionResponse(**result)
    except FeatureValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"missing_features": exc.missing, "extra_features": exc.extra},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (ModelNotReadyError, ModelArtifactError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
