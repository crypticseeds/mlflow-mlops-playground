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

### Install Dependencies
Install MLflow and Jupyter kernel support:
```bash
uv pip install mlflow ipykernel
```

Alternatively (dependency-managed):
```bash
uv add mlflow ipykernel
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
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Point MLflow APIs to your running tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# (Optional) Organize runs by experiment
mlflow.set_experiment("my_experiment")
```

#### Basic Experiment Tracking
```python
with mlflow.start_run(run_name="quick-check"):
    mlflow.log_param("alpha", 1)
    mlflow.log_metric("test", 2)
```

#### Complete ML Pipeline Example
```python
with mlflow.start_run(run_name="random-forest-experiment"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions and calculate metrics
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Log metrics
    mlflow.log_metric("mse", mse)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log additional artifacts (plots, data samples, etc.)
    mlflow.log_artifact("feature_importance.png")
```

#### Advanced Usage Patterns
```python
# Set experiment with tags
mlflow.set_experiment("house-price-prediction")
with mlflow.start_run(run_name="feature-engineering-v2"):
    # Add tags for better organization
    mlflow.set_tag("model_type", "random_forest")
    mlflow.set_tag("feature_engineering", "v2")
    mlflow.set_tag("dataset", "housing")
    
    # Log multiple metrics
    mlflow.log_metrics({
        "train_mse": train_mse,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "r2_score": r2_score
    })
    
    # Log model with custom metadata
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="HousePriceModel"
    )
```

### Learning Journey & Documentation

This section documents key learnings and experiments conducted during the MLflow exploration:

#### Experiment Tracking
- **Basic Tracking**: Started with simple parameter and metric logging
- **Experiment Organization**: Learned to use `mlflow.set_experiment()` for organizing related runs
- **Run Naming**: Discovered the importance of descriptive run names for better organization

#### Model Management
- **Model Logging**: Explored `mlflow.sklearn.log_model()` for model persistence
- **Model Registry**: Experimented with model versioning and staging
- **Artifact Storage**: Understanding how MLflow stores models and other artifacts

#### Key Learnings
1. **Tracking URI Configuration**: Always set the tracking URI when working with remote/local MLflow servers
2. **Experiment Naming**: Use consistent, descriptive experiment names
3. **Parameter Logging**: Log all hyperparameters for reproducibility
4. **Metric Tracking**: Track both training and validation metrics
5. **Artifact Management**: Save models, plots, and other outputs as artifacts

#### Project Structure
- `1-MLproject/`: Contains Jupyter notebooks with MLflow experiments
- `mlruns/`: MLflow tracking data (auto-generated)
- `mlartifacts/`: Model artifacts and metadata storage

#### Best Practices Discovered
- Always use context managers (`with mlflow.start_run()`) for automatic run management
- Log parameters before training, metrics after evaluation
- Use descriptive tags for better run organization
- Save model artifacts with proper metadata for easy retrieval

### Notes
- If the UI is already running, new runs from notebooks will appear under the configured experiment.
- Use consistent experiment names to keep runs organized.
- The `mlruns/` directory contains all tracking data and should be committed to version control for team collaboration.