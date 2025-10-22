## mlflow-mlops-playground

This repository documents my learning journey with MLflow — experiments, notes, and small examples as I explore tracking, UI, and basic MLOps workflows.

### Environment setup (uv)
- Ensure Python ≥ 3.13 is available.
- Install uv (package manager + venv):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Create and activate a project-scoped virtual environment:
```bash
uv venv
source .venv/bin/activate
```

### Install MLflow
Using uv’s pip compatibility:
```bash
uv pip install mlflow
```

Alternatively (dependency-managed):
```bash
uv add mlflow
```

### Start the MLflow UI
Run from the project root:
```bash
mlflow ui
```
By default, the UI is served at `127.0.0.1:5000`.

### Using MLflow in notebooks
Always import MLflow early in the notebook and set the tracking URI (especially if the UI is running locally):
```python
import mlflow

# Point MLflow APIs to your running tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# (Optional) Organize runs by experiment
mlflow.set_experiment("my_experiment")
```

Example minimal run:
```python
with mlflow.start_run(run_name="quick-check"):
    mlflow.log_param("alpha", 1)
    mlflow.log_metric("test", 2)
```

### Notes
- If the UI is already running, new runs from notebooks will appear under the configured experiment.
- Use consistent experiment names to keep runs organized.