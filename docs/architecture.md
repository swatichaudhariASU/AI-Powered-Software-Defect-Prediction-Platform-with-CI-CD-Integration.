# Architecture

## Month 1 data foundation

```text
GitHub repositories
       |
       +----------------------+
       |                      |
       v                      v
Git history              Source files
       |                      |
       v                      v
Process metrics          Code metrics
       |                      |
       +----------+-----------+
                  |
                  v
          Feature dataset
                  |
                  v
       Preliminary candidate labels
```

## Long-term architecture

```text
GitHub Pull Request
        |
        v
GitHub Actions
        |
        v
Metric extraction
        |
        v
FastAPI prediction service
        |
        v
ML model
   +----+-----+
   |          |
   v          v
Database    Prediction response
                |
                v
          React dashboard
```
