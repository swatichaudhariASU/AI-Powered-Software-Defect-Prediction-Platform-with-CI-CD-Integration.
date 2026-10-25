# AI-Powered Software Defect Predictor


## objective

Build the research and data foundation before model training:
1. Understand the defect prediction problem.
2. Define requirements and MVP scope.
3. Set up the repository and Python environment.
4. Clone selected GitHub repositories.
5. Extract Git history/process metrics.
6. Extract Python source-code metrics.
7. Combine metrics into a feature dataset.
8. Create preliminary candidate defect labels.
9. Perform initial data-quality analysis.

## Important research note

Keyword-based bug-fix detection creates **candidate labels**, not ground-truth defect labels.
For later research work, evaluate stronger labeling approaches such as issue-linked commits
and SZZ, and use time-aware evaluation to reduce leakage.

## Quick start

Create and activate a virtual environment, then install:

```bash
python -m pip install -r requirements.txt
```

Run the environment check:

```bash
python ml/src/environment_test.py
```

Add repository URLs to:

`data/repository_list.csv`

Then clone them:

```bash
python ml/src/clone_repositories.py
```

Extract Git history:

```bash
python ml/src/git_metrics.py
```

Extract Python code metrics:

```bash
python ml/src/code_metrics.py
```

Build combined features:

```bash
python ml/src/build_features.py
```

Create preliminary candidate labels:

```bash
python ml/src/label_data.py
```

Run data-quality analysis:

```bash
python ml/src/explore_dataset.py
```

## Scope

Month 1 intentionally does not include AWS, React, SHAP, XGBoost, PostgreSQL,
production deployment, authentication, or automated retraining.

## Backend model integration

The FastAPI backend now supports a saved-model loading contract for defect-risk prediction.

Run tests:

```bash
pytest -q
```

Start the API with the finalized research model at `ml/models/best_model.joblib`:

```bash
uvicorn backend.app.main:app --reload
```

If the finalized model has not been handed off yet, create a synthetic artifact only for local integration testing:

```bash
python ml/src/create_demo_model_artifact.py
export DEFECT_MODEL_PATH=ml/models/demo_model.joblib
uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and review `/health`, `/model/info`, and `/predict`.

The synthetic artifact is for software integration verification only and is not a research result.


## ML training and evaluation

The ML workstream includes baseline Logistic Regression and Random Forest pipelines, held-out evaluation (precision, recall, F1, PR-AUC, and ROC-AUC where defined), model comparison output, and a versioned Joblib artifact contract.

After repository metrics and preliminary labels have been generated, run:

```bash
python -m ml.src.build_features
python -m ml.src.label_data
python -m ml.src.train_models
pytest -q
```

The training script expects `data/processed/preliminary_dataset.csv`. Repository clones and generated CSVs are intentionally not included in this source ZIP. The keyword-based `candidate_defect` labels are noisy preliminary labels, not confirmed defect ground truth. See [`docs/swati_two_week_plan.md`](docs/swati_two_week_plan.md) for the ML work plan and limitations.


## Swati's October ML work: evaluation and label audit

For the October 12–16 work, the training script compares Logistic Regression and Random Forest using precision, recall, F1, PR-AUC, and ROC-AUC where defined. It writes JSON/CSV comparison reports and feature-importance data under `reports/ml/`. For October 19–23, audit candidate labels against commit-message keywords and issue references:

```bash
python -m ml.src.validate_labels
```

This creates `reports/labels/label_audit.csv` and `label_audit_summary.json`. The audit is heuristic and flags rows for review; it does not establish ground-truth defect labels. See `docs/swati_oct12_oct23_work_plan.md` for the day-by-day work split.
