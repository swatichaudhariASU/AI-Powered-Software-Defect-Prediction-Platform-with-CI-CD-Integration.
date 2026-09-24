from fastapi import FastAPI

app = FastAPI(title="AI Software Defect Predictor", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}
