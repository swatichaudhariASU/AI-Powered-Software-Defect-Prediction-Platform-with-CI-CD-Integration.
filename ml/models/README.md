# Model artifacts

Sejal's backend integration expects the finalized defect-prediction artifact at:

`ml/models/best_model.joblib`

Model binaries are intentionally ignored by Git. The backend also accepts another path through the `DEFECT_MODEL_PATH` environment variable.

## Preferred artifact contract

Save a Joblib dictionary with these keys:

```python
{
    "model": trained_model,
    "features": ["loc", "cyclomatic_complexity", ...],
    "target": "defects",
    "model_version": "2026-10-baseline-v1",
    "metadata": {"optional": "research metadata"},
}
```

The backend also accepts a raw scikit-learn estimator, but the dictionary format is preferred because it preserves the feature order and model metadata.

## Local integration demo

To verify the loading flow before the finalized research model is available:

```bash
python ml/src/create_demo_model_artifact.py
export DEFECT_MODEL_PATH=ml/models/demo_model.joblib
uvicorn backend.app.main:app --reload
```

The demo artifact uses synthetic data only for backend integration testing. Do not report its performance as a research result.
