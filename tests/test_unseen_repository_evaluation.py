"""Tests for repository-held-out evaluation and prediction-quality reports."""
import json
import numpy as np
import pandas as pd
import pytest

from ml.src.train_models import FEATURES, TARGET
from ml.src.evaluate_unseen_repositories import evaluate_unseen_repositories


def make_multi_repo_dataset(path, repositories=6, rows_per_repo=20):
    rng = np.random.default_rng(91)
    rows = []
    for repo_idx in range(repositories):
        for row_idx in range(rows_per_repo):
            label = row_idx % 2
            record = {feature: float(rng.normal(loc=label * 2, scale=1)) for feature in FEATURES}
            record.update({
                TARGET: label,
                "repository": f"repo_{repo_idx}",
                "file_path": f"src/module_{row_idx}.py",
            })
            rows.append(record)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_evaluation_holds_out_whole_repositories_and_writes_reports(tmp_path):
    dataset = tmp_path / "dataset.csv"
    report_dir = tmp_path / "reports"
    make_multi_repo_dataset(dataset)
    report = evaluate_unseen_repositories(dataset, report_dir)

    assert report["evaluation"] == "repository_group_holdout"
    assert report["unseen_test_repositories"]
    assert report["repository_overlap"] == []
    assert set(report["models"]) == {"logistic_regression", "random_forest"}
    for metrics in report["models"].values():
        assert {"precision", "recall", "f1", "pr_auc", "roc_auc"}.issubset(metrics)
    expected = [
        "unseen_repository_evaluation.json",
        "unseen_repository_model_comparison.csv",
        "unseen_repository_predictions.csv",
        "unseen_repository_feature_importance.csv",
        "unseen_repository_metrics_by_repo.csv",
    ]
    assert all((report_dir / name).exists() for name in expected)
    details = pd.read_csv(report_dir / "unseen_repository_predictions.csv")
    assert {"false_positive", "false_negative", "true_positive", "true_negative"}.issuperset(
        set(details["prediction_outcome"])
    )
    saved = json.loads((report_dir / "unseen_repository_evaluation.json").read_text())
    assert saved["repository_overlap"] == []


def test_evaluation_requires_repository_column(tmp_path):
    dataset = tmp_path / "dataset.csv"
    frame = pd.DataFrame({feature: [1, 2] * 5 for feature in FEATURES})
    frame[TARGET] = [0, 1] * 5
    frame.to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="repository"):
        evaluate_unseen_repositories(dataset, tmp_path / "reports")


def test_evaluation_requires_multiple_repositories(tmp_path):
    dataset = tmp_path / "dataset.csv"
    frame = pd.DataFrame({feature: list(range(20)) for feature in FEATURES})
    frame[TARGET] = [0, 1] * 10
    frame["repository"] = "only_repo"
    frame.to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="at least two distinct repositories"):
        evaluate_unseen_repositories(dataset, tmp_path / "reports")
