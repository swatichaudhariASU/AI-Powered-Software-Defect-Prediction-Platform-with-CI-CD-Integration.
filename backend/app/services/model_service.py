"""Service for managing the defect prediction model."""

from pathlib import Path
from typing import Any


class ModelService:
    """Manage model loading and model metadata."""

    def __init__(self) -> None:
        self.model: Any = None
        self.model_version: str | None = None
        self.model_path: Path | None = None

    @property
    def is_loaded(self) -> bool:
        """Return whether a trained model is currently loaded."""
        return self.model is not None

    def get_model_info(self) -> dict:
        """Return information about the current model."""
        return {
            "model_loaded": self.is_loaded,
            "model_version": self.model_version,
            "model_path": (
                str(self.model_path)
                if self.model_path is not None
                else None
            ),
        }


model_service = ModelService()
