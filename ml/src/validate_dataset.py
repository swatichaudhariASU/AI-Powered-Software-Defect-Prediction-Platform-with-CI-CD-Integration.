"""Validate the feature dataset before comparing baseline defect classifiers.

This utility reports schema problems, duplicate rows, non-numeric feature values,
missing feature values, invalid labels, and class balance. It does not silently
modify the source dataset or claim that heuristic labels are ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.train_models import FEATURES, TARGET, INPUT, REPORT_DIR


VALIDATION_JSON = REPORT_DIR / "dataset_validation.json"
VALIDATION_CSV = REPORT_DIR / "feature_validation.csv"


def validate_dataset(input_path: Path = INPUT) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}. Generate metrics and candidate labels first."
        )

    frame = pd.read_csv(input_path)
    required = FEATURES + [TARGET]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Dataset contains no rows.")

    selected = frame[required].copy()
    feature_rows: list[dict[str, Any]] = []
    invalid_numeric_total = 0
    for feature in FEATURES:
        original = selected[feature]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_numeric = int((original.notna() & converted.isna()).sum())
        missing_before = int(original.isna().sum())
        missing_after = int(converted.isna().sum())
        invalid_numeric_total += invalid_numeric
        feature_rows.append({
            "feature": feature,
            "dtype_in_file": str(original.dtype),
            "missing_values": missing_before,
            "invalid_numeric_values": invalid_numeric,
            "missing_or_invalid_after_conversion": missing_after,
            "unique_numeric_values": int(converted.nunique(dropna=True)),
            "all_values_missing_or_invalid": bool(converted.notna().sum() == 0),
        })
        selected[feature] = converted

    target_original = selected[TARGET]
    target_numeric = pd.to_numeric(target_original, errors="coerce")
    invalid_target_values = int((target_numeric.isna() | ~target_numeric.isin([0, 1])).sum())
    valid_target = target_numeric[target_numeric.isin([0, 1])].astype(int)
    class_counts = {str(label): int(count) for label, count in valid_target.value_counts().sort_index().items()}
    duplicate_rows = int(selected.duplicated().sum())
    feature_report = pd.DataFrame(feature_rows)

    summary: dict[str, Any] = {
        "input_file": str(input_path),
        "row_count": int(len(selected)),
        "duplicate_rows": duplicate_rows,
        "required_features": FEATURES,
        "target": TARGET,
        "invalid_numeric_feature_values": invalid_numeric_total,
        "invalid_target_values": invalid_target_values,
        "valid_labeled_rows": int(valid_target.shape[0]),
        "class_distribution": class_counts,
        "both_target_classes_present": len(class_counts) == 2,
        "features_with_missing_or_invalid_values": [
            row["feature"] for row in feature_rows
            if row["missing_or_invalid_after_conversion"] > 0
        ],
        "features_with_no_valid_numeric_values": [
            row["feature"] for row in feature_rows if row["all_values_missing_or_invalid"]
        ],
        "label_note": (
            "candidate_defect is keyword-derived and may be noisy; this report checks data quality, "
            "not label correctness or ground truth."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    feature_report.to_csv(VALIDATION_CSV, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {VALIDATION_JSON}")
    print(f"Wrote {VALIDATION_CSV}")
    return summary


if __name__ == "__main__":
    validate_dataset()
