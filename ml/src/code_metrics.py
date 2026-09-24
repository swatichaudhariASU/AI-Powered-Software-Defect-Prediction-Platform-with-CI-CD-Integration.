"""Extract basic Python source metrics using Radon."""

from pathlib import Path
import pandas as pd
from radon.complexity import cc_visit
from radon.raw import analyze

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT / "data" / "raw" / "repositories"
OUTPUT = ROOT / "data" / "raw" / "code_metrics.csv"


def analyze_file(path: Path, repo_name: str, relative_path: str):
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        raw = analyze(source)
        blocks = cc_visit(source)

        return {
            "repository": repo_name,
            "file_path": relative_path,
            "loc": int(raw.loc),
            "lloc": int(raw.lloc),
            "sloc": int(raw.sloc),
            "comments": int(raw.comments),
            "multi": int(raw.multi),
            "blank": int(raw.blank),
            "cyclomatic_complexity": float(
                sum(block.complexity for block in blocks)
            ),
            "num_functions": int(sum(
                1 for block in blocks
                if block.__class__.__name__ in {"Function", "AsyncFunction"}
            )),
            "num_classes": int(sum(
                1 for block in blocks if block.__class__.__name__ == "Class"
            )),
        }
    except Exception as exc:
        print(f"Skipping {path}: {exc}")
        return None


def main():
    rows = []

    for repo_dir in sorted(REPO_ROOT.iterdir()):
        if not repo_dir.is_dir() or not (repo_dir / ".git").exists():
            continue

        for path in repo_dir.rglob("*.py"):
            if ".git" in path.parts:
                continue
            relative = path.relative_to(repo_dir).as_posix()
            result = analyze_file(path, repo_dir.name, relative)
            if result:
                rows.append(result)

    df = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} with {len(df)} files.")


if __name__ == "__main__":
    main()
