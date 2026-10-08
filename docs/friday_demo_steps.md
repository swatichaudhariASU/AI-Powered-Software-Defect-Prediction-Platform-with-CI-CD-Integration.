# Friday Demo, October 9, 2026

## 1. Activate the environment

```bash
source .venv/bin/activate
```

## 2. Run backend tests

```bash
pytest -q
```

Save a screenshot of the passing test output.

## 3. If the finalized research model is available

Place it at `ml/models/best_model.joblib`, then start the API:

```bash
uvicorn backend.app.main:app --reload
```

## 4. If the finalized model is not available yet

Generate the clearly labeled synthetic integration artifact:

```bash
python ml/src/create_demo_model_artifact.py
export DEFECT_MODEL_PATH=ml/models/demo_model.joblib
uvicorn backend.app.main:app --reload
```

This verifies model loading and API integration only. Do not present demo-model performance as a research finding.

## 5. Open Swagger

Open `http://127.0.0.1:8000/docs`.

Show:

- `GET /health`
- `GET /model/info`
- `POST /predict`

Use `docs/sample_prediction_request.json` for the prediction request.

## 6. Evidence to keep

- `pytest -q` screenshot
- Swagger `/docs` screenshot
- `/model/info` JSON
- `/predict` JSON
- `git log --oneline -10`
- `git status`
- `docs/progress/2026-10-09-sejal.md`
