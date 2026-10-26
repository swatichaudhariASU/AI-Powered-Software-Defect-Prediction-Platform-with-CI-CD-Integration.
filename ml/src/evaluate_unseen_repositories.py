"""Evaluate baseline defect models on repositories excluded from training.

This is a group holdout evaluation: every row from a repository belongs wholly
in either train or test, preventing same-repository files from crossing the split.
Labels remain keyword-derived candidates, not confirmed defect ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from ml.src.train_models import FEATURES, TARGET, INPUT, REPORT_DIR, make_models

REPOSITORY_COLUMN = "repository"
RANDOM_STATE = 42


def _metrics(y_true: pd.Series, predicted: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    classes = set(pd.Series(y_true).dropna().astype(int).unique())
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if classes == {0, 1} else None,
        "true_negatives_false_positives_false_negatives_true_positives": [
            int(value) for value in confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        ],
    }


def _choose_repository_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Find a deterministic group split where both train and test contain both labels."""
    groups = frame[REPOSITORY_COLUMN].astype(str)
    if groups.nunique() < 2:
        raise ValueError(
            "Unseen-repository evaluation requires at least two distinct repositories. "
            "Add repositories to data/repository_list.csv and regenerate the dataset."
        )
    splitter = GroupShuffleSplit(n_splits=200, test_size=0.25, random_state=RANDOM_STATE)
    y = frame[TARGET].astype(int)
    for train_idx, test_idx in splitter.split(frame, y, groups):
        if y.iloc[train_idx].nunique() == 2 and y.iloc[test_idx].nunique() == 2:
            return train_idx, test_idx
    raise ValueError(
        "Could not find a repository-level split containing both candidate labels in train and test. "
        "Add more repositories or review class distribution. A random file-level split is not used as fallback."
    )


def evaluate_unseen_repositories(input_path: Path = INPUT, report_dir: Path = REPORT_DIR) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}. Generate metrics and labels first.")
    frame = pd.read_csv(input_path)
    required = set(FEATURES + [TARGET, REPOSITORY_COLUMN])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame = frame[FEATURES + [TARGET, REPOSITORY_COLUMN]].copy()
    frame[REPOSITORY_COLUMN] = frame[REPOSITORY_COLUMN].fillna("").astype(str).str.strip()
    frame = frame[frame[REPOSITORY_COLUMN] != ""].copy()
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.dropna(subset=[TARGET])
    frame = frame[frame[TARGET].isin([0, 1])].copy()
    frame[TARGET] = frame[TARGET].astype(int)
    if len(frame) < 10 or frame[TARGET].nunique() != 2:
        raise ValueError("Need at least 10 labeled rows and both candidate labels (0 and 1).")

    train_idx, test_idx = _choose_repository_split(frame)
    train_frame = frame.iloc[train_idx].copy()
    test_frame = frame.iloc[test_idx].copy()
    x_train, y_train = train_frame[FEATURES], train_frame[TARGET]
    x_test, y_test = test_frame[FEATURES], test_frame[TARGET]

    report_dir.mkdir(parents=True, exist_ok=True)
    comparison: dict[str, Any] = {}
    predictions_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    per_repo_rows: list[dict[str, Any]] = []
    for model_name, model in make_models().items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        predicted = (probabilities >= 0.5).astype(int)
        comparison[model_name] = _metrics(y_test, predicted, probabilities)

        for idx, (_, row) in enumerate(test_frame.iterrows()):
            actual = int(row[TARGET])
            pred = int(predicted[idx])
            predictions_rows.append({
                "model": model_name,
                "repository": row[REPOSITORY_COLUMN],
                "file_path": row.get("file_path", ""),
                "actual_candidate_defect": actual,
                "predicted_candidate_defect": pred,
                "predicted_probability": float(probabilities[idx]),
                "prediction_outcome": "true_positive" if actual == 1 and pred == 1 else
                    "false_positive" if actual == 0 and pred == 1 else
                    "false_negative" if actual == 1 and pred == 0 else "true_negative",
            })

        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
            method = "random_forest_impurity_importance"
        else:
            values = np.abs(estimator.coef_[0])
            method = "absolute_logistic_regression_coefficient"
        for feature, value in zip(FEATURES, values):
            importance_rows.append({
                "model": model_name,
                "feature": feature,
                "importance": float(value),
                "importance_method": method,
                "interpretation_note": "Association/importance, not causal evidence; LR coefficients are absolute scaled coefficients.",
            })

        model_preds = pd.DataFrame([
            row for row in predictions_rows if row["model"] == model_name
        ])
        for repo, repo_frame in model_preds.groupby("repository"):
            actual = repo_frame["actual_candidate_defect"].astype(int)
            pred = repo_frame["predicted_candidate_defect"].astype(int)
            prob = repo_frame["predicted_probability"].astype(float)
            per_repo_rows.append({
                "model": model_name,
                "repository": repo,
                **_metrics(actual, pred.to_numpy(), prob.to_numpy()),
            })

    test_repos = sorted(test_frame[REPOSITORY_COLUMN].unique().tolist())
    train_repos = sorted(train_frame[REPOSITORY_COLUMN].unique().tolist())
    report: dict[str, Any] = {
        "evaluation": "repository_group_holdout",
        "input_file": str(input_path),
        "row_count_after_validation": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_repositories": train_repos,
        "unseen_test_repositories": test_repos,
        "repository_overlap": sorted(set(train_repos) & set(test_repos)),
        "features": FEATURES,
        "target": TARGET,
        "decision_threshold": 0.5,
        "random_state": RANDOM_STATE,
        "models": comparison,
        "interpretation_warning": (
            "candidate_defect labels are keyword-derived and noisy; this is preliminary evaluation, "
            "not proof of performance against confirmed defect ground truth. Repository-level results "
            "can vary with repository selection and dataset size."
        ),
    }
    (report_dir / "unseen_repository_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame.from_dict(comparison, orient="index").rename_axis("model").reset_index().to_csv(
        report_dir / "unseen_repository_model_comparison.csv", index=False
    )
    pd.DataFrame(predictions_rows).to_csv(report_dir / "unseen_repository_predictions.csv", index=False)
    pd.DataFrame(importance_rows).sort_values(["model", "importance"], ascending=[True, False]).to_csv(
        report_dir / "unseen_repository_feature_importance.csv", index=False
    )
    pd.DataFrame(per_repo_rows).to_csv(report_dir / "unseen_repository_metrics_by_repo.csv", index=False)
    print(json.dumps(report, indent=2))
    print(f"Wrote unseen-repository reports to {report_dir}")
    return report


if __name__ == "__main__":
    evaluate_unseen_repositories()
