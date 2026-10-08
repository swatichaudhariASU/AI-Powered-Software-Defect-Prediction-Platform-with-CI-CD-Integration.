"""Tests for the FastAPI backend and model-loading flow."""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from backend.app.main import app
from backend.app.services.model_service import DEFAULT_FEATURES, ModelService, model_service

client = TestClient(app)


def valid_payload() -> dict[str, float | int]:
    return {
        "loc": 250,
        "cyclomatic_complexity": 12,
        "function_count": 8,
        "class_count": 2,
        "commit_count": 24,
        "author_count": 5,
        "lines_added": 480,
        "lines_deleted": 210,
        "churn": 690,
    }


def build_test_artifact(path: Path) -> None:
    rows = [
        [80, 3, 4, 1, 5, 1, 90, 20, 110],
        [120, 5, 5, 1, 7, 2, 130, 35, 165],
        [180, 8, 7, 2, 12, 3, 220, 75, 295],
        [230, 11, 9, 2, 20, 4, 400, 180, 580],
        [350, 16, 13, 4, 32, 7, 700, 310, 1010],
        [500, 24, 18, 6, 48, 9, 1100, 600, 1700],
    ]
    labels = [0, 0, 0, 1, 1, 1]
    frame = pd.DataFrame(rows, columns=DEFAULT_FEATURES)
    model = LogisticRegression(max_iter=500, random_state=42).fit(frame, labels)
    joblib.dump(
        {
            "model": model,
            "features": DEFAULT_FEATURES,
            "target": "defects",
            "model_version": "test-model-v1",
        },
        path,
    )


@pytest.fixture
def loaded_test_model(tmp_path, monkeypatch):
    artifact_path = tmp_path / "test_model.joblib"
    build_test_artifact(artifact_path)
    monkeypatch.setattr(model_service, "model_path", artifact_path)
    monkeypatch.setattr(model_service, "_artifact", None)
    model_service.load(required=True)
    yield artifact_path


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert isinstance(response.json()["model_loaded"], bool)


def test_model_info_endpoint_exposes_feature_contract():
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["features"] == DEFAULT_FEATURES
    assert body["target"] == "defects"


def test_model_service_loads_saved_artifact(tmp_path):
    artifact_path = tmp_path / "saved_model.joblib"
    build_test_artifact(artifact_path)
    service = ModelService(model_path=artifact_path)
    assert service.load(required=True) is True
    assert service.is_loaded is True
    assert service.model_version == "test-model-v1"
    assert service.features == DEFAULT_FEATURES


def test_prediction_endpoint_returns_model_output(loaded_test_model):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "prediction_complete"
    assert body["defect_prediction"] in [0, 1]
    assert 0.0 <= body["defect_probability"] <= 1.0
    assert body["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert body["model_version"] == "test-model-v1"


def test_prediction_endpoint_returns_503_without_model(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing.joblib"
    monkeypatch.setattr(model_service, "model_path", missing_path)
    monkeypatch.setattr(model_service, "_artifact", None)
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 503
    assert "Model artifact not found" in response.json()["detail"]


def test_prediction_endpoint_rejects_negative_metrics():
    payload = valid_payload()
    payload["loc"] = -1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_endpoint_rejects_unknown_metrics():
    payload = valid_payload()
    payload["unknown_metric"] = 1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
