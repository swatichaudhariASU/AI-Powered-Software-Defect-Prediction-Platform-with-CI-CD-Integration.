"""Request and response schemas for defect prediction API."""

from pydantic import BaseModel, ConfigDict, Field


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


class PredictionResponse(BaseModel):
    """Defect-risk prediction returned by the model service."""

    status: str
    defect_prediction: int
    defect_probability: float
    risk_level: str
    model_name: str | None
    model_version: str | None


class ModelInfoResponse(BaseModel):
    """Current model-loading status and feature contract."""

    model_loaded: bool
    model_name: str | None
    model_version: str | None
    target: str
    features: list[str]
    model_path: str
