"""Create a synthetic model artifact for local API integration testing only."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "loc",
    "cyclomatic_complexity",
    "function_count",
    "class_count",
    "commit_count",
    "author_count",
    "lines_added",
    "lines_deleted",
    "churn",
]


def build_demo_artifact(output_path: Path) -> Path:
    rows = [
        [70, 2, 3, 1, 4, 1, 70, 10, 80],
        [95, 4, 4, 1, 6, 2, 110, 25, 135],
        [130, 5, 5, 1, 8, 2, 150, 40, 190],
        [175, 7, 7, 2, 11, 3, 220, 70, 290],
        [210, 9, 8, 2, 15, 3, 300, 110, 410],
        [250, 12, 10, 3, 22, 5, 480, 210, 690],
        [310, 14, 12, 3, 28, 6, 600, 260, 860],
        [390, 18, 15, 4, 35, 7, 780, 360, 1140],
        [470, 22, 18, 5, 43, 8, 980, 500, 1480],
        [560, 28, 21, 6, 52, 10, 1250, 690, 1940],
    ]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    frame = pd.DataFrame(rows, columns=FEATURES)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    model.fit(frame, labels)

    artifact = {
        "model": model,
        "features": FEATURES,
        "target": "defects",
        "model_version": "synthetic-integration-demo-v1",
        "metadata": {
            "purpose": "API integration testing only",
            "research_result": False,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    path = build_demo_artifact(project_root / "ml" / "models" / "demo_model.joblib")
    print(f"Created demo artifact: {path}")
    print("Synthetic integration artifact only. Do not report its metrics as research results.")
