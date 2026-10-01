"""Request and response schemas for defect prediction API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Software metrics required for defect-risk prediction."""

    loc: int = Field(ge=0)
    cyclomatic_complexity: float = Field(ge=0)
    function_count: int = Field(ge=0)
    class_count: int = Field(ge=0)
    commit_count: int = Field(ge=0)
    author_count: int = Field(ge=0)
    lines_added: int = Field(ge=0)
    lines_deleted: int = Field(ge=0)
    churn: int = Field(ge=0)


class PredictionResponse(BaseModel):
    """Response returned by the prediction API."""

    status: str
    defect_probability: float | None = None
    risk_level: str
    model_version: str | None = None
