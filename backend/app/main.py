"""FastAPI application entry point."""

from fastapi import FastAPI

from backend.app.api.predictions import router as prediction_router
from backend.app.services.model_service import model_service

app = FastAPI(
    title="AI-Powered Software Defect Prediction API",
    version="0.2.0",
    description=(
        "Backend service for validating software metrics, loading a saved "
        "defect-prediction model, and serving defect-risk predictions."
    ),
)

app.include_router(prediction_router)


@app.get("/health", tags=["system"])
def health() -> dict:
    """Return API health plus current model-loading state."""
    info = model_service.get_model_info()
    return {
        "status": "ok",
        "model_loaded": info["model_loaded"],
        "model_version": info["model_version"],
    }
