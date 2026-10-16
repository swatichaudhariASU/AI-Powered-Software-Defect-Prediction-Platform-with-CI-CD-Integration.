"""Model loading and inference helpers for defect-risk prediction."""

from __future__ import annotations

import math
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

DEFAULT_MEDIUM_RISK_THRESHOLD = 0.40
DEFAULT_HIGH_RISK_THRESHOLD = 0.70


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
        self.medium_risk_threshold = self._read_threshold(
            "DEFECT_MEDIUM_RISK_THRESHOLD", DEFAULT_MEDIUM_RISK_THRESHOLD
        )
        self.high_risk_threshold = self._read_threshold(
            "DEFECT_HIGH_RISK_THRESHOLD", DEFAULT_HIGH_RISK_THRESHOLD
        )
        if self.medium_risk_threshold >= self.high_risk_threshold:
            raise ValueError(
                "DEFECT_MEDIUM_RISK_THRESHOLD must be lower than "
                "DEFECT_HIGH_RISK_THRESHOLD."
            )
        self._artifact: dict[str, Any] | None = None

    @staticmethod
    def _read_threshold(name: str, default: float) -> float:
        value = float(os.getenv(name, default))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
        return value

    @property
    def is_loaded(self) -> bool:
        return self._artifact is not None

    def unload(self) -> None:
        """Clear the in-memory artifact without deleting the saved model file."""
        self._artifact = None

    def load(self, required: bool = False) -> bool:
        """Load the configured model artifact.

        Preferred artifact format:
        model, features, target, model_version, and optional metadata.
        A raw scikit-learn estimator remains supported for compatibility.
        """
        if not self.model_path.exists():
            self._artifact = None
            if required:
                raise ModelNotReadyError(
                    f"Model artifact not found at {self.model_path}. "
                    "Set DEFECT_MODEL_PATH or add the finalized artifact."
                )
            return False

        try:
            saved = joblib.load(self.model_path)
        except Exception as exc:
            self._artifact = None
            raise ModelArtifactError(
                f"Unable to load model artifact at {self.model_path}: {exc}"
            ) from exc

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

    @property
    def metadata(self) -> dict[str, Any]:
        if self.is_loaded:
            return dict(self._artifact.get("metadata") or {})
        return {}

    def model_name(self) -> str | None:
        if not self.is_loaded:
            return None
        model = self._artifact["model"]
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            return type(model.named_steps["model"]).__name__
        return type(model).__name__

    def risk_thresholds(self) -> dict[str, float]:
        return {
            "medium": self.medium_risk_threshold,
            "high": self.high_risk_threshold,
        }

    def get_model_info(self) -> dict[str, Any]:
        """Return model status without failing when the artifact is not present."""
        if not self.is_loaded and self.model_path.exists():
            try:
                self.load(required=False)
            except (ModelArtifactError, ValueError, TypeError):
                self._artifact = None

        evaluation_metrics = self.metadata.get("evaluation_metrics")
        if evaluation_metrics is not None and not isinstance(evaluation_metrics, dict):
            evaluation_metrics = None

        return {
            "model_loaded": self.is_loaded,
            "model_name": self.model_name(),
            "model_version": self.model_version,
            "target": self.target,
            "features": self.features,
            "model_path": str(self.model_path),
            "risk_thresholds": self.risk_thresholds(),
            "evaluation_metrics": evaluation_metrics,
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

        non_finite = sorted(
            name for name, value in metrics.items()
            if isinstance(value, float) and not math.isfinite(value)
        )
        if non_finite:
            raise ValueError(
                "Metrics must be finite numbers. Invalid metrics: "
                + ", ".join(non_finite)
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
        risk_level = self.risk_level(probability)

        return {
            "status": "prediction_complete",
            "defect_prediction": prediction,
            "defect_probability": round(probability, 4),
            "risk_level": risk_level,
            "model_name": self.model_name(),
            "model_version": self.model_version,
            "feature_count": len(self.features),
            "risk_thresholds": self.risk_thresholds(),
        }

    def risk_level(self, probability: float) -> str:
        """Convert a defect probability into LOW, MEDIUM, or HIGH risk."""
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Defect probability must be between 0 and 1.")
        if probability >= self.high_risk_threshold:
            return "HIGH"
        if probability >= self.medium_risk_threshold:
            return "MEDIUM"
        return "LOW"

    def _defect_probability(self, row: pd.DataFrame, prediction: int) -> float:
        if not hasattr(self.model, "predict_proba"):
            return float(prediction)

        probabilities = self.model.predict_proba(row)[0]
        classes = list(getattr(self.model, "classes_", []))
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])
        elif len(probabilities) == 2:
            probability = float(probabilities[1])
        else:
            probability = float(max(probabilities))

        if not math.isfinite(probability):
            raise ModelArtifactError("Model returned a non-finite probability.")
        return min(max(probability, 0.0), 1.0)


model_service = ModelService()
