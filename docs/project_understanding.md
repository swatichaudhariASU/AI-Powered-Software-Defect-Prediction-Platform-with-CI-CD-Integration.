# Project Understanding

## Problem

Software defect prediction uses historical repository information and software metrics
to estimate which files or changes are more likely to be associated with defects.

## Proposed system

The project will eventually analyze changed files in a pull request, calculate metrics,
send those metrics to a trained ML model, and report a risk score through CI/CD.

## Initial prediction unit

The Month 1 dataset is file-oriented. A later iteration can move toward change-level
or just-in-time prediction.

## Important limitation

A risk score is a probability/ranking signal, not proof that a file contains a defect.
