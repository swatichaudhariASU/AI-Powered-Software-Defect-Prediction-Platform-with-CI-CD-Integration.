"""Train and compare baseline Logistic Regression and Random Forest models.

Input: data/processed/preliminary_dataset.csv with candidate_defect labels.
Output: evaluation reports and best_model.joblib. Candidate keyword labels are
noisy research labels and must not be presented as confirmed defects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "preliminary_dataset.csv"
REPORT_DIR = ROOT / "reports" / "ml"
MODEL_DIR = ROOT / "ml" / "models"
FEATURES = [
    "loc", "cyclomatic_complexity", "function_count", "class_count",
    "commit_count", "author_count", "lines_added", "lines_deleted", "churn",
]
TARGET = "candidate_defect"
RANDOM_STATE = 42


def make_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }


def metric_summary(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    result: dict[str, Any] = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    }
    result["roc_auc"] = (
        float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None
    )
    return result


def train(input_path: Path = INPUT) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}. Run metric extraction and labeling first."
        )
    frame = pd.read_csv(input_path)
    required = set(FEATURES + [TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame = frame[FEATURES + [TARGET]].copy()
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.dropna(subset=[TARGET])
    frame = frame[frame[TARGET].isin([0, 1])]
    frame[TARGET] = frame[TARGET].astype(int)
    if len(frame) < 10 or frame[TARGET].nunique() != 2:
        raise ValueError(
            "Need at least 10 labeled rows and both classes (0 and 1) to train. "
            f"Rows={len(frame)}, labels={frame[TARGET].value_counts().to_dict()}"
        )
    if frame[TARGET].value_counts().min() < 2:
        raise ValueError("Need at least two examples of each class for a stratified split.")

    x = frame[FEATURES]
    y = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in make_models().items():
        model.fit(x_train, y_train)
        fitted[name] = model
        results[name] = metric_summary(model, x_test, y_test)

    # PR-AUC is the primary comparison metric for the imbalanced-label problem.
    winner = max(results, key=lambda name: results[name]["pr_auc"])
    report = {
        "input_file": str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else str(input_path),
        "row_count": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "features": FEATURES,
        "target": TARGET,
        "label_note": "candidate_defect is keyword-derived and may contain noisy labels",
        "split": {"method": "stratified random holdout", "test_size": 0.25, "random_state": RANDOM_STATE},
        "primary_selection_metric": "pr_auc",
        "models": results,
        "selected_model": winner,
    }
    (REPORT_DIR / "model_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    artifact = {
        "model": fitted[winner],
        "features": FEATURES,
        "target": TARGET,
        "model_version": f"{winner}-candidate-labels-v1",
        "metadata": {
            "selected_by": "holdout PR-AUC",
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "label_source": "candidate_defect keyword heuristic; not ground truth",
            "evaluation_metrics": results[winner],
        },
    }
    joblib.dump(artifact, MODEL_DIR / "best_model.joblib")
    (REPORT_DIR / "selected_model.txt").write_text(
        f"Selected: {winner}\nSelection metric: PR-AUC\n"
        "Warning: metrics use preliminary keyword-derived labels; validate labels before research claims.\n"
    )
    print(json.dumps(report, indent=2))
    print(f"Saved model artifact: {MODEL_DIR / 'best_model.joblib'}")
    return report


if __name__ == "__main__":
    train()
