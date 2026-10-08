# Git Commit Plan for October 9

The current branch is `sejal/backend-prediction-api`.

Before committing:

```bash
git status
git diff
pytest -q tests/test_backend_api.py
```

## Commit 1, model loading service and ML contract

```bash
git add backend/app/services/model_service.py ml/models/README.md ml/src/create_demo_model_artifact.py docs/model_artifact_contract.md .gitignore
git commit -m "Implement model loading flow and artifact contract"
```

## Commit 2, prediction API integration

```bash
git add backend/app/api/predictions.py backend/app/main.py backend/app/schemas/prediction.py docs/sample_prediction_request.json
git commit -m "Connect prediction API to model service"
```

## Commit 3, tests and Friday documentation

```bash
git add tests/test_backend_api.py README.md docs/friday_demo_steps.md docs/progress/2026-10-09-sejal.md docs/git_commit_plan_oct9.md evidence/backend_verification_2026-10-08.txt
git commit -m "Add backend integration tests and Friday progress evidence"
```

Then verify:

```bash
git log --oneline -10
git status
git push origin sejal/backend-prediction-api
```

Do not commit `ml/models/demo_model.joblib`. Model binaries remain ignored by Git.
