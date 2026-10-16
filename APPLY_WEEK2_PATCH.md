# Apply Week 2 Patch

This patch is for Sejal's `sejal/backend-prediction-api` branch for Oct 12-16, 2026.

## What changes

- completes and hardens `POST /predict`
- adds probability-based LOW / MEDIUM / HIGH risk output
- adds configurable risk thresholds
- validates missing, extra, negative, and non-finite metrics
- improves missing/corrupt model error handling
- exposes optional model evaluation metrics through `GET /model/info`
- expands backend API tests to 10 cases
- adds Week 2 documentation and evidence

## Apply from the repository root

Preview first:

```bash
rsync -avn /path/to/Sejal_Week2_Oct16_Patch/ ./
```

Apply:

```bash
rsync -av /path/to/Sejal_Week2_Oct16_Patch/ ./
```

Then run:

```bash
source .venv/bin/activate
pytest -q tests/test_backend_api.py
```

Expected result for this patch:

```text
10 passed
```

Review `docs/git_commit_plan_oct16.md` for the suggested commit sequence.
