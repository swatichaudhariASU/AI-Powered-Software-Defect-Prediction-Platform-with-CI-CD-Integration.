# Swati — Week 4: Unseen-Repository Evaluation and Prediction Quality

## Objective
Evaluate baseline Logistic Regression and Random Forest on repository/repositories that are completely excluded from training; inspect per-file predictions, false positives/negatives, and feature importance.

## Implementation
Run `python -m ml.src.evaluate_unseen_repositories` after generating `data/processed/preliminary_dataset.csv`. The script uses `repository` as the group key and uses a group holdout split. It will not silently fall back to a random file-level split if the data cannot support a valid repository split.

## Outputs
- `reports/ml/unseen_repository_evaluation.json`: split metadata, repository lists, overlap check, model metrics, and caveats.
- `reports/ml/unseen_repository_model_comparison.csv`: LR/RF summary metrics.
- `reports/ml/unseen_repository_predictions.csv`: held-out predictions, probability, actual candidate label, and TP/FP/FN/TN category.
- `reports/ml/unseen_repository_feature_importance.csv`: Random Forest impurity importance and absolute LR coefficient values.
- `reports/ml/unseen_repository_metrics_by_repo.csv`: metrics for each held-out repository and model.

## How to interpret
- Check that `repository_overlap` is empty. A non-empty overlap would invalidate the unseen-repository claim.
- Review precision, recall, F1, PR-AUC and ROC-AUC when defined. Small or imbalanced held-out repositories can make metrics unstable.
- Inspect false positives and false negatives in the prediction CSV and compare them with the label audit from Week 3.
- Feature importance is descriptive, not causal. LR uses absolute coefficient magnitude; RF uses impurity-based importance, which has known biases.
- `candidate_defect` is generated from commit-message keywords. Metrics are preliminary and are not performance against verified defect ground truth.
- Do not use the held-out test set to repeatedly tune thresholds or choose a final model. Use it as a final generalization check; keep threshold tuning separate.

## Commands
```bash
python -m ml.src.evaluate_unseen_repositories
pytest -q tests/test_unseen_repository_evaluation.py
```

## Data prerequisite
`data/processed/preliminary_dataset.csv` must include `repository`, all nine model features, and `candidate_defect`. At least two repositories and both labels must be represented in a valid train/test group split. If `data/repository_list.csv` is empty or cloned repositories have not been processed, add repositories, run the data pipeline, and rerun evaluation.
