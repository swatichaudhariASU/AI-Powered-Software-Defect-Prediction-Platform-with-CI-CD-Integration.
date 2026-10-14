"""Combine source-code and Git process metrics into the model feature contract."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "data" / "raw" / "code_metrics.csv"
GIT = ROOT / "data" / "raw" / "git_history.csv"
OUTPUT = ROOT / "data" / "processed" / "features.csv"


def main() -> None:
    if not CODE.exists() or not GIT.exists():
        missing = [str(path) for path in (CODE, GIT) if not path.exists()]
        raise FileNotFoundError(
            "Run the repository metric extractors first. Missing input(s): "
            + ", ".join(missing)
        )

    code = pd.read_csv(CODE)
    git = pd.read_csv(GIT)
    keys = ["repository", "file_path"]
    if not set(keys).issubset(code.columns) or not set(keys).issubset(git.columns):
        raise ValueError("Both input files must contain repository and file_path columns.")

    df = code.merge(git, on=keys, how="left", suffixes=("", "_git"))
    aliases = {"num_functions": "function_count", "num_classes": "class_count"}
    for source, target in aliases.items():
        if source in df.columns and target not in df.columns:
            df[target] = df[source]

    numeric = [
        "loc", "cyclomatic_complexity", "function_count", "class_count",
        "commit_count", "author_count", "lines_added", "lines_deleted", "churn",
    ]
    for col in numeric:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} with {len(df)} rows and the model feature contract.")


if __name__ == "__main__":
    main()
