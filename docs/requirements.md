# Requirements

## Functional requirements

- FR-01: Analyze repositories containing Git history.
- FR-02: Extract source-code metrics from Python files.
- FR-03: Extract file-level Git history/process metrics.
- FR-04: Build a reproducible feature dataset.
- FR-05: Create preliminary candidate defect labels.
- FR-06: Train and evaluate ML models in a later phase.
- FR-07: Analyze changed files in pull requests in a later phase.
- FR-08: Report risk through GitHub Actions in a later phase.
- FR-09: Provide prediction history through a dashboard in a later phase.

## Non-functional requirements

- Reproducibility
- Clear separation of data collection, ML, backend, CI, and frontend components
- Secure handling of credentials
- Graceful handling of missing or unsupported source files
- Testable Python modules
- Time-aware evaluation for historical defect data

## Month 1 scope

Requirements, architecture, repository mining, metric extraction, preliminary labeling,
and dataset quality analysis.
