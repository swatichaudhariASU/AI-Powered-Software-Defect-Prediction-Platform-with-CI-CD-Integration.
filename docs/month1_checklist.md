# Month 1 Build Checklist

## Week 1 — Understanding and requirements
- [ ] Read and summarize defect-prediction fundamentals.
- [ ] Review relevant advisor/research papers.
- [ ] Write problem statement.
- [ ] Write functional and non-functional requirements.
- [ ] Define MVP and future scope.

## Week 2 — Architecture and setup
- [ ] Draw architecture.
- [ ] Confirm technology stack.
- [ ] Create Git repository.
- [ ] Create project folders.
- [ ] Create Python virtual environment.
- [ ] Install requirements.
- [ ] Run environment test.
- [ ] Run FastAPI health endpoint.

## Week 3 — Repository mining
- [ ] Choose 3–5 suitable Python repositories.
- [ ] Add URLs to data/repository_list.csv.
- [ ] Clone repositories.
- [ ] Inspect git log manually.
- [ ] Run git_metrics.py.
- [ ] Inspect generated git_history.csv.

## Week 4 — Code metrics and preliminary labels
- [ ] Run code_metrics.py.
- [ ] Inspect code_metrics.csv.
- [ ] Run build_features.py.
- [ ] Inspect features.csv.
- [ ] Run label_data.py.
- [ ] Run explore_dataset.py.
- [ ] Document missing values and class distribution.
- [ ] Commit the Month 1 implementation and documentation.

## Important
Do not call keyword-derived labels ground truth. They are preliminary candidate labels.
Do not train/test a final model on a random split of historical data; later use time-aware
splits and carefully document the labeling methodology.
