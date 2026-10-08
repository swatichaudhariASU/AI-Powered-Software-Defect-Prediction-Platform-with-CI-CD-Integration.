"""Model loading and inference helpers for defect-risk prediction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


DEFAULT_FEATURES = [
    "loc",
    "cyclomatic_complexity",
    "function_count",
    "class_count",
    "commit_count",
    "author_count",
    "lines_added",
    "lines_deleted",
    "churn",
]


class ModelNotReadyError(RuntimeError):
    """Raised when prediction is requested before a model is available."""


class ModelArtifactError(RuntimeError):
    """Raised when a saved model artifact has an unsupported structure."""


class FeatureValidationError(ValueError):
    """Raised when supplied metrics do not match the model feature contract."""

    def __init__(self, missing: list[str], extra: list[str]) -> None:
        self.missing = missing
        self.extra = extra
        super().__init__(f"Missing features: {missing}; extra features: {extra}")


class ModelService:
    """Load a saved model artifact and expose metadata and prediction helpers."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        configured_path = os.getenv("DEFECT_MODEL_PATH")
        self.model_path = Path(
            model_path
            or configured_path
            or project_root / "ml" / "models" / "best_model.joblib"
        )
        self._artifact: dict[str, Any] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._artifact is not None

    def unload(self) -> None:
        """Clear the in-memory artifact without deleting the saved model file."""
        self._artifact = None

    def load(self, required: bool = False) -> bool:
        """Load the configured model artifact.

        The preferred artifact format is a dictionary containing:
        model, features, target, and optional model_version metadata.
        A raw scikit-learn estimator is also accepted for compatibility.
        """
        if not self.model_path.exists():
            self._artifact = None
            if required:
                raise ModelNotReadyError(
                    f"Model artifact not found at {self.model_path}. "
                    "Set DEFECT_MODEL_PATH or add the finalized artifact."
                )
            return False

        saved = joblib.load(self.model_path)

        if isinstance(saved, dict) and "model" in saved:
            model = saved["model"]
            features = list(saved.get("features") or self._infer_features(model))
            target = str(saved.get("target") or "defects")
            model_version = str(saved.get("model_version") or self.model_path.stem)
            metadata = dict(saved.get("metadata") or {})
        else:
            model = saved
            features = self._infer_features(model)
            target = "defects"
            model_version = self.model_path.stem
            metadata = {}

        if not hasattr(model, "predict"):
            raise ModelArtifactError("Saved artifact does not expose a predict() method.")

        if not features:
            raise ModelArtifactError(
                "No feature order was found in the model artifact. "
                "Store a 'features' list with the model artifact."
            )

        self._artifact = {
            "model": model,
            "features": features,
            "target": target,
            "model_version": model_version,
            "metadata": metadata,
        }
        return True

    @staticmethod
    def _infer_features(model: Any) -> list[str]:
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            return [str(name) for name in feature_names]
        return list(DEFAULT_FEATURES)

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load(required=True)

    @property
    def model(self) -> Any:
        self._ensure_loaded()
        return self._artifact["model"]

    @property
    def features(self) -> list[str]:
        if self.is_loaded:
            return list(self._artifact["features"])
        return list(DEFAULT_FEATURES)

    @property
    def target(self) -> str:
        if self.is_loaded:
            return str(self._artifact["target"])
        return "defects"

    @property
    def model_version(self) -> str | None:
        if self.is_loaded:
            return str(self._artifact["model_version"])
        return None

    def model_name(self) -> str | None:
        if not self.is_loaded:
            return None
        model = self._artifact["model"]
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            return type(model.named_steps["model"]).__name__
        return type(model).__name__

    def get_model_info(self) -> dict[str, Any]:
        """Return model status without failing when the artifact is not present."""
        if not self.is_loaded and self.model_path.exists():
            try:
                self.load(required=False)
            except (ModelArtifactError, ValueError, TypeError):
                self._artifact = None

        return {
            "model_loaded": self.is_loaded,
            "model_name": self.model_name(),
            "model_version": self.model_version,
            "target": self.target,
            "features": self.features,
            "model_path": str(self.model_path),
        }

    def validate_metrics(self, metrics: dict[str, float]) -> None:
        expected = set(self.features)
        supplied = set(metrics)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            raise FeatureValidationError(missing=missing, extra=extra)

        negative = sorted(name for name, value in metrics.items() if value < 0)
        if negative:
            raise ValueError(
                "Metrics must be non-negative. Invalid metrics: " + ", ".join(negative)
            )

    def predict(self, metrics: dict[str, float]) -> dict[str, Any]:
        """Return a defect prediction using the configured saved model."""
        self._ensure_loaded()
        self.validate_metrics(metrics)

        row = pd.DataFrame(
            [[metrics[name] for name in self.features]],
            columns=self.features,
        )

        prediction = int(self.model.predict(row)[0])
        probability = self._defect_probability(row=row, prediction=prediction)

        if probability >= 0.70:
            risk_level = "HIGH"
        elif probability >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "status": "prediction_complete",
            "defect_prediction": prediction,
            "defect_probability": round(probability, 4),
            "risk_level": risk_level,
            "model_name": self.model_name(),
            "model_version": self.model_version,
        }

    def _defect_probability(self, row: pd.DataFrame, prediction: int) -> float:
        if not hasattr(self.model, "predict_proba"):
            return float(prediction)

        probabilities = self.model.predict_proba(row)[0]
        classes = list(getattr(self.model, "classes_", []))
        if 1 in classes:
            return float(probabilities[classes.index(1)])
        if len(probabilities) == 2:
            return float(probabilities[1])
        return float(max(probabilities))


model_service = ModelService()
