# Problem Statement

Develop an AI-powered software defect prediction system that analyzes source-code and
repository-history metrics to estimate defect risk for files changed during software
development.

The long-term system will integrate the prediction into a GitHub Actions pre-merge
workflow and expose prediction history through a web dashboard.

## Inputs

- Source-code metrics
- Git history/process metrics
- Historical defect labels

## Output

A model-generated defect-risk probability or score for each analyzed file.

## Long-term workflow

GitHub PR -> changed files -> metric extraction -> ML inference -> risk report -> CI/CD gate
