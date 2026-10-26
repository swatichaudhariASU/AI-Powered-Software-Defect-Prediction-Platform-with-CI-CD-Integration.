"""Tests for evidence-based candidate defect-label auditing."""
import pandas as pd
import pytest

from ml.src import audit_defect_labels


def test_audit_flags_issue_linked_keyword_and_keyword_only_rows(tmp_path):
    commits_path = tmp_path / "git_commit_files.csv"
    dataset_path = tmp_path / "preliminary_dataset.csv"
    report_dir = tmp_path / "reports"
    pd.DataFrame([
        {"repository": "repo-a", "file_path": "src/a.py", "commit_hash": "abc1", "commit_message": "Fix crash, closes #123"},
        {"repository": "repo-a", "file_path": "src/b.py", "commit_hash": "abc2", "commit_message": "bug fix for parsing"},
        {"repository": "repo-a", "file_path": "src/c.py", "commit_hash": "abc3", "commit_message": "Refactor and see #456"},
    ]).to_csv(commits_path, index=False)
    pd.DataFrame([
        {"repository": "repo-a", "file_path": "src/a.py", "candidate_defect": 1},
        {"repository": "repo-a", "file_path": "src/b.py", "candidate_defect": 1},
        {"repository": "repo-a", "file_path": "src/c.py", "candidate_defect": 0},
        {"repository": "repo-a", "file_path": "src/d.py", "candidate_defect": 0},
    ]).to_csv(dataset_path, index=False)

    summary = audit_defect_labels.audit_labels(commits_path, dataset_path, report_dir)
    audit = pd.read_csv(report_dir / "label_audit.csv")
    by_path = audit.set_index("file_path")
    assert by_path.loc["src/a.py", "audit_category"] == "positive_with_keyword_and_issue_reference"
    assert by_path.loc["src/b.py", "audit_category"] == "positive_keyword_without_issue_reference"
    assert by_path.loc["src/c.py", "audit_category"] == "negative_with_issue_reference_only"
    assert by_path.loc["src/d.py", "audit_category"] == "no_keyword_or_issue_reference_evidence"
    assert (audit["review_status"] == "pending_manual_review").all()
    assert summary["dataset_rows_audited"] == 4
    assert (report_dir / "label_audit_summary.json").exists()


def test_audit_requires_expected_columns(tmp_path):
    commits_path = tmp_path / "commits.csv"
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame({"repository": ["r"]}).to_csv(commits_path, index=False)
    pd.DataFrame({"repository": ["r"], "file_path": ["x.py"], "candidate_defect": [0]}).to_csv(dataset_path, index=False)
    with pytest.raises(ValueError, match="Commit evidence is missing required columns"):
        audit_defect_labels.audit_labels(commits_path, dataset_path, tmp_path / "reports")


def test_audit_reports_missing_inputs_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Commit evidence not found"):
        audit_defect_labels.audit_labels(tmp_path / "missing.csv", tmp_path / "dataset.csv", tmp_path / "reports")
