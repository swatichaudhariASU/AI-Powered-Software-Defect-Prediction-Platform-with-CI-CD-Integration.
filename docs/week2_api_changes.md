# Week 2 Prediction API Changes

## API behavior

`POST /predict` now returns:

- binary defect prediction
- defect probability between 0 and 1
- `LOW`, `MEDIUM`, or `HIGH` risk level
- model name and version
- number of model features
- active risk thresholds

## Default risk thresholds

- LOW: probability below 0.40
- MEDIUM: probability from 0.40 through values below 0.70
- HIGH: probability 0.70 or greater

Thresholds can be changed through:

```bash
export DEFECT_MEDIUM_RISK_THRESHOLD=0.40
export DEFECT_HIGH_RISK_THRESHOLD=0.70
```

## Validation

The API rejects:

- missing required metrics
- unknown metrics
- negative metric values
- non-finite numeric values
- requests when the model artifact is missing or unreadable

## Model evaluation metadata

If the finalized model artifact contains `metadata.evaluation_metrics`, `GET /model/info` returns those values. This allows the backend to expose the validated model-comparison results without embedding research numbers in source code.
