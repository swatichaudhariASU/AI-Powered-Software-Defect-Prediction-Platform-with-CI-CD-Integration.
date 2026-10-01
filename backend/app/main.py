"""FastAPI application entry point."""

from fastapi import FastAPI

from backend.app.api.predictions import router as prediction_router
from backend.app.services.model_service import model_service

app = FastAPI(
    title="AI Software Defect Predictor",
    version="0.1.0",
    description="API for software quality analysis and AI-powered defect-risk prediction.",
)

app.include_router(prediction_router)


@app.get("/health", tags=["system"])
def health():
    """Return application health status."""
    return {"status": "ok"}


@app.get("/model-info", tags=["system"])
def model_info():
    """Return information about the current prediction model."""
    return model_service.get_model_info()
