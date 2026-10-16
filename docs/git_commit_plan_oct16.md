# Week 2 Git Commit Plan

Run these commits after reviewing and testing the files.

```bash
git add backend/app/services/model_service.py backend/app/schemas/prediction.py
git commit -m "Strengthen prediction validation and risk scoring"
```

```bash
git add backend/app/api/predictions.py backend/app/main.py
git commit -m "Finalize prediction API responses and error handling"
```

```bash
git add tests/test_backend_api.py
git commit -m "Expand prediction API and risk threshold tests"
```

```bash
git add docs/progress/2026-10-16-sejal.md docs/week2_api_changes.md docs/git_commit_plan_oct16.md evidence/week2_backend_verification_2026-10-16.txt
git commit -m "Document Week 2 prediction API progress"
```

Then review and push:

```bash
git log --oneline -10
git status
git push origin sejal/backend-prediction-api
```
