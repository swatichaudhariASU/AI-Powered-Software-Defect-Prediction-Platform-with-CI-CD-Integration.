"""Clone Python repositories listed in data/repository_list.csv."""

from pathlib import Path
import sys
import pandas as pd
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

ROOT = Path(__file__).resolve().parents[2]
LIST_FILE = ROOT / "data" / "repository_list.csv"
DEST = ROOT / "data" / "raw" / "repositories"


def main():
    DEST.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(LIST_FILE, comment="#")
    except pd.errors.EmptyDataError:
        print("Repository list is empty. Add repositories to data/repository_list.csv.")
        return

    required = {"repository", "url", "language"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    if df.empty:
        print("Repository list contains no repositories yet.")
        return

    for row in df.itertuples(index=False):
        if str(row.language).strip().lower() != "python":
            print(f"Skipping {row.repository}: only Python is supported in Month 1.")
            continue

        target = DEST / str(row.repository)
        if (target / ".git").exists():
            print(f"Already cloned: {row.repository}")
            continue

        try:
            print(f"Cloning {row.repository}...")
            Repo.clone_from(str(row.url), target)
            print(f"Cloned: {row.repository}")
        except (GitCommandError, InvalidGitRepositoryError) as exc:
            print(f"Failed to clone {row.repository}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
