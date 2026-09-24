"""Combine source-code and Git process metrics."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "data" / "raw" / "code_metrics.csv"
GIT = ROOT / "data" / "raw" / "git_history.csv"
OUTPUT = ROOT / "data" / "processed" / "features.csv"


def main():
    code = pd.read_csv(CODE)
    git = pd.read_csv(GIT)

    keys = ["repository", "file_path"]
    df = code.merge(git, on=keys, how="left")

    numeric = [
        "commit_count", "author_count", "lines_added",
        "lines_deleted", "churn"
    ]
    for col in numeric:
        df[col] = df[col].fillna(0)

    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} with {len(df)} rows.")


if __name__ == "__main__":
    main()
