"""Audit preliminary defect labels against commit-message and issue-reference evidence.

This is an evidence triage tool, not an automatic ground-truth labeler. It flags
files/commits for manual review and never silently changes candidate_defect labels.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COMMITS_INPUT = ROOT / "data" / "raw" / "git_commit_files.csv"
DATASET_INPUT = ROOT / "data" / "processed" / "preliminary_dataset.csv"
REPORT_DIR = ROOT / "reports" / "labels"
AUDIT_CSV = REPORT_DIR / "label_audit.csv"
SUMMARY_JSON = REPORT_DIR / "label_audit_summary.json"

KEYWORD_PATTERN = re.compile(
    r"\b(fix|fixed|fixes|bug|bugs|defect|error|issue|patch|hotfix|regression)\b",
    re.IGNORECASE,
)
# Covers common GitHub issue/PR references, e.g. #123, owner/repo#123,
# github.com/owner/repo/issues/123, and github.com/owner/repo/pull/123.
ISSUE_REFERENCE_PATTERN = re.compile(
    r"(?<![\w/])#\d+\b|[\w.-]+/[\w.-]+#\d+|"
    r"github\.com/[^\s/]+/[^\s/]+/(?:issues|pull)/\d+",
    re.IGNORECASE,
)
REQUIRED_COMMIT_COLUMNS = {"repository", "file_path", "commit_hash", "commit_message"}
REQUIRED_DATASET_COLUMNS = {"repository", "file_path", "candidate_defect"}


def _has(pattern: re.Pattern[str], value: Any) -> bool:
    return bool(pattern.search(str(value or "")))


def audit_labels(
    commits_path: Path = COMMITS_INPUT,
    dataset_path: Path = DATASET_INPUT,
    report_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    """Create file-level evidence audit and summary without changing model labels."""
    if not commits_path.exists():
        raise FileNotFoundError(
            f"Commit evidence not found: {commits_path}. Run ml.src.git_metrics first."
        )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found: {dataset_path}. Run metric extraction, feature building, and labeling first."
        )

    commits = pd.read_csv(commits_path)
    dataset = pd.read_csv(dataset_path)
    missing_commits = sorted(REQUIRED_COMMIT_COLUMNS - set(commits.columns))
    missing_dataset = sorted(REQUIRED_DATASET_COLUMNS - set(dataset.columns))
    if missing_commits:
        raise ValueError(f"Commit evidence is missing required columns: {missing_commits}")
    if missing_dataset:
        raise ValueError(f"Dataset is missing required columns: {missing_dataset}")

    commits = commits.copy()
    commits["commit_message"] = commits["commit_message"].fillna("").astype(str)
    commits["has_defect_keyword"] = commits["commit_message"].map(lambda value: _has(KEYWORD_PATTERN, value))
    commits["has_issue_reference"] = commits["commit_message"].map(lambda value: _has(ISSUE_REFERENCE_PATTERN, value))
    commits["has_both_signals"] = commits["has_defect_keyword"] & commits["has_issue_reference"]

    rows: list[dict[str, Any]] = []
    for _, row in dataset.iterrows():
        repository, file_path = row["repository"], row["file_path"]
        file_commits = commits[
            (commits["repository"].astype(str) == str(repository))
            & (commits["file_path"].astype(str) == str(file_path))
        ]
        keyword_commits = file_commits[file_commits["has_defect_keyword"]]
        issue_commits = file_commits[file_commits["has_issue_reference"]]
        both_commits = file_commits[file_commits["has_both_signals"]]
        label_value = pd.to_numeric(pd.Series([row["candidate_defect"]]), errors="coerce").iloc[0]
        valid_label = pd.notna(label_value) and label_value in (0, 1)
        candidate_label = int(label_value) if valid_label else None

        if candidate_label == 1 and len(both_commits):
            category = "positive_with_keyword_and_issue_reference"
            reason = "Candidate positive has at least one commit message with a defect keyword and issue/PR reference."
        elif candidate_label == 1 and len(keyword_commits):
            category = "positive_keyword_without_issue_reference"
            reason = "Candidate positive is supported by a keyword commit, but no matching keyword commit has an issue/PR reference. Review manually."
        elif candidate_label == 1:
            category = "positive_without_matching_keyword_evidence"
            reason = "Dataset is positive but no matching keyword commit was found in the supplied commit evidence. Check data freshness and join keys."
        elif candidate_label == 0 and len(both_commits):
            category = "negative_with_keyword_and_issue_reference"
            reason = "Dataset is negative despite a matching commit with both signals. Check label generation and review manually."
        elif candidate_label == 0 and len(issue_commits):
            category = "negative_with_issue_reference_only"
            reason = "A matching commit references an issue/PR but lacks a defect keyword; this is not proof of a defect fix. Review manually."
        elif candidate_label == 0 and len(keyword_commits):
            category = "negative_with_keyword_evidence"
            reason = "Dataset is negative despite a matching defect-keyword commit. Check label generation and review manually."
        elif candidate_label in (0, 1):
            category = "no_keyword_or_issue_reference_evidence"
            reason = "No matching commit message with the configured keyword/issue-reference patterns was found. This is not proof that the label is correct or false."
        else:
            category = "invalid_candidate_label"
            reason = "Candidate label is not 0 or 1; inspect the dataset row."

        rows.append({
            "repository": repository,
            "file_path": file_path,
            "candidate_defect": candidate_label,
            "audit_category": category,
            "matching_commit_count": int(len(file_commits)),
            "keyword_commit_count": int(len(keyword_commits)),
            "issue_reference_commit_count": int(len(issue_commits)),
            "keyword_and_issue_commit_count": int(len(both_commits)),
            "keyword_commit_hashes": ";".join(keyword_commits["commit_hash"].dropna().astype(str).unique()),
            "issue_reference_commit_hashes": ";".join(issue_commits["commit_hash"].dropna().astype(str).unique()),
            "evidence_messages": " || ".join(file_commits.loc[
                file_commits["has_defect_keyword"] | file_commits["has_issue_reference"], "commit_message"
            ].dropna().astype(str).drop_duplicates().head(10)),
            "review_status": "pending_manual_review",
            "review_notes": "",
            "audit_reason": reason,
        })

    audit = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(report_dir / "label_audit.csv", index=False)
    category_counts = {str(k): int(v) for k, v in audit["audit_category"].value_counts().sort_index().items()}
    summary: dict[str, Any] = {
        "commit_evidence_file": str(commits_path),
        "candidate_dataset_file": str(dataset_path),
        "dataset_rows_audited": int(len(audit)),
        "commit_file_records_reviewed": int(len(commits)),
        "rows_pending_manual_review": int((audit["review_status"] == "pending_manual_review").sum()),
        "audit_category_counts": category_counts,
        "issue_reference_definition": "GitHub-style #number, owner/repo#number, or GitHub issues/pull URL patterns in commit messages.",
        "limitations": [
            "An issue/PR reference does not prove that a commit fixes a defect.",
            "Commit-message keyword matches can be false positives; defects may be fixed without keywords.",
            "Files changed by a fix are not necessarily the files containing the root cause.",
            "This script flags evidence for manual review and does not rewrite candidate_defect labels.",
            "Issue references are detected from available commit messages only; this script does not fetch issue tracker contents or validate issue status.",
        ],
    }
    (report_dir / "label_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {report_dir / 'label_audit.csv'}")
    print(f"Wrote {report_dir / 'label_audit_summary.json'}")
    return summary


if __name__ == "__main__":
    audit_labels()
