"""Create preliminary candidate defect labels from commit-message keywords."""

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_COMMITS = ROOT / "data" / "raw" / "git_commit_files.csv"
FEATURES = ROOT / "data" / "processed" / "features.csv"
OUTPUT = ROOT / "data" / "processed" / "preliminary_dataset.csv"

PATTERN = re.compile(
    r"\b(fix|fixed|fixes|bug|bugs|defect|error|issue|patch|hotfix)\b",
    re.IGNORECASE,
)


def main():
    commits = pd.read_csv(RAW_COMMITS)
    features = pd.read_csv(FEATURES)

    if commits.empty:
        features["candidate_defect"] = 0
        features.to_csv(OUTPUT, index=False)
        print("No Git commit data available; wrote all-zero preliminary labels.")
        return

    candidates = commits[
        commits["commit_message"].fillna("").map(lambda x: bool(PATTERN.search(str(x))))
    ]

    candidate_files = (
        candidates[["repository", "file_path"]]
        .drop_duplicates()
        .assign(candidate_defect=1)
    )

    result = features.merge(candidate_files, on=["repository", "file_path"], how="left")
    result["candidate_defect"] = result["candidate_defect"].fillna(0).astype(int)

    result.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
