# 5MLET-Datathon - Passos Mágicos (Defasagem)

Predicting student lag ("defasagem") for the Passos Mágicos NGO project.

## Project Structure
- `src/passos_magicos`: Core logic for preprocessing, model dispatching, and training.
- `notebooks`: Exploratory analysis and prototyping.
- `config.yaml`: Central configuration for experiments.

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Training Pipeline
To train a model using the configuration in `config.yaml`:
```bash
# Make sure you are in the project root
uv run python -m src.passos_magicos.train
```

**What happens when you run this?**
- Loads parameters from `config.yaml`.
- Preprocesses data (cleaning, feature engineering).
- Trains the model (e.g., Random Forest, XGBoost).
- Logs metrics, parameters, and artifacts to MLflow.
- Registers the model as `passos_magicos_defasagem_v1`.

### 3. View Results (MLflow UI)
To visualize experiments, compare runs, and see artifacts:
```bash
uv run mlflow ui
```
Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Configuration
Edit `config.yaml` to change:
- **Model Type**: `random_forest`, `xgboost`, `lightgbm`.
- **Hyperparameters**: `n_estimators`, `max_depth`, etc.
- **Features**: Add/remove features in `features` list.