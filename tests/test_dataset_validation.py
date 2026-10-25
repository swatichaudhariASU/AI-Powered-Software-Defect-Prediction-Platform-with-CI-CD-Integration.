"""Tests for Week 2 dataset schema and feature-quality validation."""
import json
import numpy as np
import pandas as pd
import pytest

from ml.src import validate_dataset
from ml.src.train_models import FEATURES, TARGET


def sample_frame(rows=12):
    rng = np.random.default_rng(24)
    frame = pd.DataFrame({feature: rng.integers(1, 20, rows) for feature in FEATURES})
    frame[TARGET] = [0, 1] * (rows // 2)
    return frame


def test_validation_reports_schema_missing_values_and_class_balance(tmp_path, monkeypatch):
    input_path = tmp_path / "dataset.csv"
    frame = sample_frame()
    frame.loc[0, "loc"] = np.nan
    frame["churn"] = frame["churn"].astype(object)
    frame.loc[1, "churn"] = "not-a-number"
    frame.to_csv(input_path, index=False)
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(validate_dataset, "REPORT_DIR", report_dir)
    monkeypatch.setattr(validate_dataset, "VALIDATION_JSON", report_dir / "dataset_validation.json")
    monkeypatch.setattr(validate_dataset, "VALIDATION_CSV", report_dir / "feature_validation.csv")

    result = validate_dataset.validate_dataset(input_path)

    assert result["row_count"] == 12
    assert result["invalid_numeric_feature_values"] == 1
    assert result["class_distribution"] == {"0": 6, "1": 6}
    assert "loc" in result["features_with_missing_or_invalid_values"]
    assert "churn" in result["features_with_missing_or_invalid_values"]
    assert (report_dir / "dataset_validation.json").exists()
    assert (report_dir / "feature_validation.csv").exists()
    saved = json.loads((report_dir / "dataset_validation.json").read_text())
    assert saved["target"] == TARGET


def test_validation_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({TARGET: [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset.validate_dataset(path)


def test_validation_reports_invalid_target_values(tmp_path, monkeypatch):
    input_path = tmp_path / "bad-labels.csv"
    frame = sample_frame()
    frame.loc[0, TARGET] = 2
    frame[TARGET] = frame[TARGET].astype(object)
    frame.loc[1, TARGET] = "unknown"
    frame.to_csv(input_path, index=False)
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(validate_dataset, "REPORT_DIR", report_dir)
    monkeypatch.setattr(validate_dataset, "VALIDATION_JSON", report_dir / "dataset_validation.json")
    monkeypatch.setattr(validate_dataset, "VALIDATION_CSV", report_dir / "feature_validation.csv")

    result = validate_dataset.validate_dataset(input_path)
    assert result["invalid_target_values"] == 2
    assert result["valid_labeled_rows"] == 10
