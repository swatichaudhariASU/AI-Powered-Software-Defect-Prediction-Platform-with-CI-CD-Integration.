# Preliminary Labeling Strategy

Month 1 uses a simple keyword-based approach only to create **candidate** bug-fix commits.

Candidate commit messages may contain terms such as:
- fix
- bug
- defect
- error
- issue
- patch

The files changed by candidate commits receive a preliminary positive association.

## Limitations

- Commit messages can be ambiguous.
- A commit containing "fix" may not fix a defect.
- A defect can be fixed without using an obvious keyword.
- Files changed by a bug fix are not necessarily the files where the defect originated.

Therefore these labels should not be described as ground truth. Later work should compare
issue-linked labeling and SZZ-style approaches and document label noise.


## Week 3 evidence audit

Run `python -m ml.src.audit_defect_labels` after generating `git_commit_files.csv`
and `preliminary_dataset.csv`. The tool flags candidate labels using commit-message
keyword and issue/PR-reference evidence, then writes `reports/labels/label_audit.csv`
and `reports/labels/label_audit_summary.json`. It does not automatically relabel files.
Human reviewers must inspect linked issue/PR details and commit diffs before documenting
false positives or missed positives. See `docs/swati_week3_label_validation.md`.
