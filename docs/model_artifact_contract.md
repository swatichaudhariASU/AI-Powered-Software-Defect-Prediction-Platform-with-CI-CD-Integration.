# Model Artifact Contract

## Purpose

The FastAPI backend needs a stable contract between the ML pipeline and the prediction service. This document defines the minimum information required when the trained model is handed off for integration.

## Expected location

Default:

`ml/models/best_model.joblib`

Alternative:

Set `DEFECT_MODEL_PATH` to another Joblib artifact path.

## Required model capabilities

The saved model must expose `predict()`.

For probability-based risk levels, the model should also expose `predict_proba()`.

## Preferred saved structure

```python
{
    "model": trained_model,
    "features": [
        "loc",
        "cyclomatic_complexity",
        "function_count",
        "class_count",
        "commit_count",
        "author_count",
        "lines_added",
        "lines_deleted",
        "churn",
    ],
    "target": "defects",
    "model_version": "<version>",
    "metadata": {},
}
```

## Backend behavior

1. `/model/info` reports whether a model is loaded and shows the expected feature order.
2. `/predict` validates incoming software metrics.
3. The service orders the values to match the training feature order.
4. The saved estimator returns a class prediction.
5. When `predict_proba()` exists, the probability for defect class `1` is returned.
6. The API maps probability to LOW, MEDIUM, or HIGH risk for integration testing.

Risk thresholds are an integration convention for now and should be reviewed with the research evaluation before final reporting.
