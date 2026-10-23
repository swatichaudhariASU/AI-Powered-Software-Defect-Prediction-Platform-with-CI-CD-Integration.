# Sejal Khedekar Project Handoff

## Project

AI-Powered Software Defect Prediction Platform with CI/CD Integration

## Git Branch

sejal/backend-prediction-api

## Completed Work

- FastAPI backend setup
- Saved model loading
- Model information endpoint
- Prediction request and response schemas
- POST /predict
- Input validation
- API error handling
- Defect probability output
- LOW, MEDIUM, and HIGH risk output
- Backend API tests
- Code-metrics integration
- Git-history metrics integration
- POST /predict/file
- Single-file defect prediction
- Single-file integration tests

## Main Endpoints

GET /health

GET /model/info

POST /predict

POST /predict/file

## Run the Backend

source .venv/bin/activate

pip install -r requirements.txt

uvicorn backend.app.main:app --reload

Swagger:

http://127.0.0.1:8000/docs

## Run Tests

pytest -q

## Single-File Prediction Flow

Python Source File
-> Code Metrics
-> Git Metrics
-> Model Features
-> ML Model
-> Defect Probability
-> Risk Level
-> FastAPI Response

## Remaining Work

- Repository-level multi-file prediction
- Repository risk aggregation
- GitHub Actions integration
- Pull-request defect prediction
- Automated CI risk reports
- React dashboard
- Prediction history database
- Docker setup
- Deployment

## Handoff Notes

The backend work is available on:

sejal/backend-prediction-api

Future development should continue from the current prediction and metrics integration flow.
