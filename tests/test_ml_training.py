"""Tests for the model training and evaluation workflow."""

import numpy as np
import pandas as pd
import pytest

from ml.src.train_models import FEATURES, TARGET, train


def make_dataset(path, rows=40):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({feature: rng.integers(0, 100, rows) for feature in FEATURES})
    frame[TARGET] = [0, 1] * (rows // 2)
    frame.to_csv(path, index=False)
    return frame


def test_training_compares_models_and_saves_artifact(tmp_path, monkeypatch):
    input_path = tmp_path / "preliminary_dataset.csv"
    make_dataset(input_path)
    report_dir = tmp_path / "reports"
    model_dir = tmp_path / "models"
    monkeypatch.setattr("ml.src.train_models.REPORT_DIR", report_dir)
    monkeypatch.setattr("ml.src.train_models.MODEL_DIR", model_dir)

    report = train(input_path)

    assert set(report["models"]) == {"logistic_regression", "random_forest"}
    assert report["selected_model"] in report["models"]
    expected_metrics = {"precision", "recall", "f1", "pr_auc", "roc_auc"}
    for model_metrics in report["models"].values():
        assert expected_metrics.issubset(model_metrics)
        for metric in ("precision", "recall", "f1", "pr_auc"):
            assert 0.0 <= model_metrics[metric] <= 1.0
    assert (report_dir / "model_comparison.json").exists()
    comparison_csv = pd.read_csv(report_dir / "model_comparison.csv")
    assert set(comparison_csv["model"]) == {"logistic_regression", "random_forest"}
    assert expected_metrics.issubset(comparison_csv.columns)
    importance_csv = pd.read_csv(report_dir / "feature_importance.csv")
    assert set(importance_csv["model"]) == {"logistic_regression", "random_forest"}
    assert (model_dir / "best_model.joblib").exists()


def test_training_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"candidate_defect": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        train(path)


def test_training_rejects_single_class(tmp_path):
    path = tmp_path / "one-class.csv"
    frame = pd.DataFrame({feature: [1] * 12 for feature in FEATURES})
    frame[TARGET] = [0] * 12
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="both classes"):
        train(path)
