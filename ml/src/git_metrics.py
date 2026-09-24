"""Extract file-level Git history metrics from cloned repositories."""

from pathlib import Path
import pandas as pd
from git import Repo

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT / "data" / "raw" / "repositories"
OUTPUT = ROOT / "data" / "raw" / "git_history.csv"
RAW_OUTPUT = ROOT / "data" / "raw" / "git_commit_files.csv"


def extract_repo(repo_path: Path, repo_name: str):
    repo = Repo(repo_path)
    records = []

    for commit in repo.iter_commits("--all"):
        try:
            stats = commit.stats.files
            for file_path, stat in stats.items():
                records.append({
                    "repository": repo_name,
                    "commit_hash": commit.hexsha,
                    "author": commit.author.email or commit.author.name,
                    "timestamp": commit.committed_datetime.isoformat(),
                    "commit_message": commit.message.strip().replace("\n", " "),
                    "file_path": file_path,
                    "lines_added": int(stat.get("insertions", 0)),
                    "lines_deleted": int(stat.get("deletions", 0)),
                })
        except Exception as exc:
            print(f"Skipping commit {commit.hexsha[:8]}: {exc}")

    return records


def aggregate(records):
    columns = [
        "repository", "file_path", "commit_count", "author_count",
        "lines_added", "lines_deleted", "churn", "first_seen", "last_seen"
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    grouped = (
        df.groupby(["repository", "file_path"], as_index=False)
        .agg(
            commit_count=("commit_hash", "nunique"),
            author_count=("author", "nunique"),
            lines_added=("lines_added", "sum"),
            lines_deleted=("lines_deleted", "sum"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
    )
    grouped["churn"] = grouped["lines_added"] + grouped["lines_deleted"]
    return grouped


def main():
    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    all_records = []

    for repo_dir in sorted(REPO_ROOT.iterdir()):
        if repo_dir.is_dir() and (repo_dir / ".git").exists():
            print(f"Processing Git history: {repo_dir.name}")
            all_records.extend(extract_repo(repo_dir, repo_dir.name))

    pd.DataFrame(all_records).to_csv(RAW_OUTPUT, index=False)
    aggregate(all_records).to_csv(OUTPUT, index=False)
    print(f"Wrote raw history: {RAW_OUTPUT}")
    print(f"Wrote file-level metrics: {OUTPUT}")


if __name__ == "__main__":
    main()
