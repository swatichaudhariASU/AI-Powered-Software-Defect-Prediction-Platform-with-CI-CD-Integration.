"""Tests for the FastAPI backend."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()

    assert body["model_loaded"] is False
    assert body["model_version"] is None
    assert body["model_path"] is None


def test_prediction_endpoint_accepts_valid_metrics():
    payload = {
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "pending_model_integration"
    assert body["defect_probability"] is None
    assert body["risk_level"] == "NOT_AVAILABLE"
    assert body["model_version"] is None


def test_prediction_endpoint_rejects_negative_metrics():
    payload = {
        "loc": -1,
        "cyclomatic_complexity": 12,
        "function_count": 8,
        "class_count": 2,
        "commit_count": 24,
        "author_count": 5,
        "lines_added": 480,
        "lines_deleted": 210,
        "churn": 690,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
