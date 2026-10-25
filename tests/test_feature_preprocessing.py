"""Tests for feature-building numeric conversion and missing-value policy."""

import pandas as pd
import pytest

from ml.src import build_features


def run_build(tmp_path, monkeypatch, code_frame, git_frame):
    code_path = tmp_path / "code_metrics.csv"
    git_path = tmp_path / "git_history.csv"
    output_path = tmp_path / "features.csv"
    code_frame.to_csv(code_path, index=False)
    git_frame.to_csv(git_path, index=False)
    monkeypatch.setattr(build_features, "CODE", code_path)
    monkeypatch.setattr(build_features, "GIT", git_path)
    monkeypatch.setattr(build_features, "OUTPUT", output_path)
    build_features.main()
    return pd.read_csv(output_path)


def test_numeric_strings_are_converted_and_aliases_are_used(tmp_path, monkeypatch):
    code = pd.DataFrame({
        "repository": ["repo-a"],
        "file_path": ["src/a.py"],
        "loc": ["12"],
        "cyclomatic_complexity": ["3"],
        "num_functions": ["2"],
        "num_classes": ["1"],
    })
    git = pd.DataFrame({
        "repository": ["repo-a"],
        "file_path": ["src/a.py"],
        "commit_count": ["4"],
        "author_count": ["2"],
        "lines_added": ["9"],
        "lines_deleted": ["5"],
        "churn": ["14"],
    })

    result = run_build(tmp_path, monkeypatch, code, git)

    assert result.loc[0, "loc"] == 12
    assert result.loc[0, "function_count"] == 2
    assert result.loc[0, "class_count"] == 1
    assert result.loc[0, "commit_count"] == 4


def test_missing_and_invalid_metrics_follow_current_zero_fill_policy(tmp_path, monkeypatch):
    code = pd.DataFrame({
        "repository": ["repo-a"],
        "file_path": ["src/a.py"],
        "loc": [None],
        "cyclomatic_complexity": ["not-a-number"],
        "function_count": [None],
        "class_count": [1],
    })
    git = pd.DataFrame({
        "repository": ["repo-a"],
        "file_path": ["src/a.py"],
    })

    result = run_build(tmp_path, monkeypatch, code, git)

    for column in [
        "loc", "cyclomatic_complexity", "function_count", "class_count",
        "commit_count", "author_count", "lines_added", "lines_deleted", "churn",
    ]:
        assert pd.api.types.is_numeric_dtype(result[column])
    for column in [
        "loc", "cyclomatic_complexity", "function_count", "commit_count",
        "author_count", "lines_added", "lines_deleted", "churn",
    ]:
        assert result.loc[0, column] == 0
    assert result.loc[0, "class_count"] == 1


def test_missing_join_keys_raise_clear_error(tmp_path, monkeypatch):
    code = pd.DataFrame({"repository": ["repo-a"], "loc": [10]})
    git = pd.DataFrame({"repository": ["repo-a"], "file_path": ["src/a.py"]})
    code_path = tmp_path / "code_metrics.csv"
    git_path = tmp_path / "git_history.csv"
    code.to_csv(code_path, index=False)
    git.to_csv(git_path, index=False)
    monkeypatch.setattr(build_features, "CODE", code_path)
    monkeypatch.setattr(build_features, "GIT", git_path)
    monkeypatch.setattr(build_features, "OUTPUT", tmp_path / "features.csv")

    with pytest.raises(ValueError, match="repository and file_path"):
        build_features.main()
