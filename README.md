# AI-Powered Software Defect Predictor

Month 1 foundation for a 10-month software defect prediction project.

## Month 1 objective

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
