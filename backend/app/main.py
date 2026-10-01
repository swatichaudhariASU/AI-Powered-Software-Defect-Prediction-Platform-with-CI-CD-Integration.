"""FastAPI application entry point."""

from fastapi import FastAPI

from backend.app.api.predictions import router as prediction_router

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
