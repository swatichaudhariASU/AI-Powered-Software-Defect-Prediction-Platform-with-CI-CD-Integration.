"""Prediction API routes."""

from fastapi import APIRouter, HTTPException

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
    """Return current model status and the expected feature order."""
    return ModelInfoResponse(**model_service.get_model_info())


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict software defect risk",
)
def predict_defect_risk(request: PredictionRequest) -> PredictionResponse:
    """Run defect-risk inference using the configured saved model artifact."""
    try:
        result = model_service.predict(request.model_dump())
        return PredictionResponse(**result)
    except FeatureValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": exc.missing, "extra_features": exc.extra},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ModelNotReadyError, ModelArtifactError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
