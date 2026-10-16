"""Request and response schemas for defect prediction API."""

from math import isfinite
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionRequest(BaseModel):
    """Software metrics required by the current backend feature contract."""

    model_config = ConfigDict(extra="forbid")

    loc: int = Field(ge=0)
    cyclomatic_complexity: float = Field(ge=0)
    function_count: int = Field(ge=0)
    class_count: int = Field(ge=0)
    commit_count: int = Field(ge=0)
    author_count: int = Field(ge=0)
    lines_added: int = Field(ge=0)
    lines_deleted: int = Field(ge=0)
    churn: int = Field(ge=0)

    @model_validator(mode="after")
    def reject_non_finite_values(self) -> "PredictionRequest":
        """Reject NaN and infinite numeric values before model inference."""
        values = self.model_dump()
        invalid = [
            name for name, value in values.items()
            if isinstance(value, float) and not isfinite(value)
        ]
        if invalid:
            raise ValueError(
                "Metrics must be finite numbers. Invalid metrics: " + ", ".join(invalid)
            )
        return self


class PredictionResponse(BaseModel):
    """Defect-risk prediction returned by the model service."""

    status: str
    defect_prediction: int
    defect_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str
    model_name: str | None
    model_version: str | None
    feature_count: int
    risk_thresholds: dict[str, float]


class ModelInfoResponse(BaseModel):
    """Current model-loading status and feature contract."""

    model_loaded: bool
    model_name: str | None
    model_version: str | None
    target: str
    features: list[str]
    model_path: str
    risk_thresholds: dict[str, float]
    evaluation_metrics: dict[str, Any] | None = None
