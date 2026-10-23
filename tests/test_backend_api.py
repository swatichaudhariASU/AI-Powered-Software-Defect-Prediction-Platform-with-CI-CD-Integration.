"""Tests for the FastAPI prediction API, validation, and risk output."""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from git import Repo
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
            "model_version": "test-model-v2",
            "metadata": {
                "evaluation_metrics": {
                    "precision": 0.80,
                    "recall": 0.75,
                    "f1": 0.77,
                    "pr_auc": 0.82,
                }
            },
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
    model_service.unload()


def test_health_endpoint_includes_thresholds():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)
    assert body["risk_thresholds"] == {"medium": 0.4, "high": 0.7}


def test_model_info_endpoint_exposes_feature_contract_and_metrics(loaded_test_model):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["features"] == DEFAULT_FEATURES
    assert body["target"] == "defects"
    assert body["model_version"] == "test-model-v2"
    assert body["risk_thresholds"] == {"medium": 0.4, "high": 0.7}
    assert body["evaluation_metrics"]["f1"] == 0.77
    assert body["evaluation_metrics"]["pr_auc"] == 0.82


def test_model_service_loads_saved_artifact(tmp_path):
    artifact_path = tmp_path / "saved_model.joblib"
    build_test_artifact(artifact_path)
    service = ModelService(model_path=artifact_path)
    assert service.load(required=True) is True
    assert service.is_loaded is True
    assert service.model_version == "test-model-v2"
    assert service.features == DEFAULT_FEATURES


def test_prediction_endpoint_returns_probability_and_risk(loaded_test_model):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "prediction_complete"
    assert body["defect_prediction"] in [0, 1]
    assert 0.0 <= body["defect_probability"] <= 1.0
    assert body["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert body["model_version"] == "test-model-v2"
    assert body["feature_count"] == len(DEFAULT_FEATURES)
    assert body["risk_thresholds"] == {"medium": 0.4, "high": 0.7}


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


def test_prediction_endpoint_rejects_missing_metric():
    payload = valid_payload()
    payload.pop("churn")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_endpoint_rejects_non_finite_metric():
    payload = valid_payload()
    payload["cyclomatic_complexity"] = "NaN"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_risk_level_boundaries(tmp_path):
    service = ModelService(model_path=tmp_path / "unused.joblib")
    assert service.risk_level(0.0) == "LOW"
    assert service.risk_level(0.3999) == "LOW"
    assert service.risk_level(0.40) == "MEDIUM"
    assert service.risk_level(0.6999) == "MEDIUM"
    assert service.risk_level(0.70) == "HIGH"
    assert service.risk_level(1.0) == "HIGH"



def create_test_git_repo(tmp_path: Path) -> Path:
    """Create a small Git repository containing one Python file."""
    repo_path = tmp_path / "sample_repo"
    repo_path.mkdir()

    repo = Repo.init(repo_path)

    with repo.config_writer() as config:
        config.set_value("user", "name", "Test User")
        config.set_value("user", "email", "test@example.com")

    source_file = repo_path / "sample.py"
    source_file.write_text(
        """
def calculate_total(values):
    total = 0
    for value in values:
        if value > 0:
            total += value
    return total
""".strip()
    )

    repo.index.add(["sample.py"])
    repo.index.commit("Add sample Python file")

    return source_file


def test_file_prediction_endpoint_returns_prediction(
    tmp_path,
    loaded_test_model,
):
    source_file = create_test_git_repo(tmp_path)

    response = client.post(
        "/predict/file",
        json={"file_path": str(source_file)},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["file_path"] == str(source_file)
    assert "metrics" in body
    assert body["metrics"]["loc"] > 0
    assert "cyclomatic_complexity" in body["metrics"]
    assert "commit_count" in body["metrics"]

    assert body["status"] == "prediction_complete"
    assert body["defect_prediction"] in [0, 1]
    assert 0.0 <= body["defect_probability"] <= 1.0
    assert body["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_file_prediction_endpoint_returns_404_for_missing_file(
    loaded_test_model,
):
    response = client.post(
        "/predict/file",
        json={
            "file_path": "/tmp/file_that_does_not_exist_12345.py"
        },
    )

    assert response.status_code == 404


def test_file_prediction_endpoint_rejects_non_python_file(
    tmp_path,
    loaded_test_model,
):
    repo_path = tmp_path / "text_repo"
    repo_path.mkdir()

    repo = Repo.init(repo_path)

    with repo.config_writer() as config:
        config.set_value("user", "name", "Test User")
        config.set_value("user", "email", "test@example.com")

    text_file = repo_path / "README.md"
    text_file.write_text("Test repository")

    repo.index.add(["README.md"])
    repo.index.commit("Add README")

    response = client.post(
        "/predict/file",
        json={"file_path": str(text_file)},
    )

    assert response.status_code == 422
