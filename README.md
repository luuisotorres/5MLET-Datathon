# 5MLET-Datathon - Passos Mágicos (Student Lagging Prediction)

## 1. Project Overview

### Objective
This project aims to predict the academic lagging risk ("defasagem") for students supported by the **Passos Mágicos Association**. By identifying students at risk early, the NGO can implement targeted psychosocial and pedagogical interventions.

### Model Output & Categories
The model prioritizes students by classifying them into one of three distinct risk statuses:
- **Critical (Severe Lagging)**: Students with high probability of significant academic delay.
- **Alert (Lagging Risk)**: Students showing early warning signs of falling behind.
- **Expected Performance**: Students meeting or exceeding academic expectations for their level.

### Proposed Solution
A complete MLOps pipeline designed to handle student data from ingestion to real-time inference:
- **Medallion Data Pipeline**: Structured data processing through Landing, Bronze (cleaning), Silver (feature engineering & drift detection), and Gold (modeling) layers.
- **MLOps Lifecycle**: Experiment tracking, parameter logging, and model versioning using **MLflow**.
- **Inference Service**: A production-ready **FastAPI** application serving predictions integrated with an **SQLite Feature Store**.
- **Observability**: Real-time service monitoring via **Prometheus** and **Grafana**, with data distribution monitoring through **Evidently AI**.

### Tech Stack
- **Python Version**: `>= 3.13`
- **ML Frameworks**: `scikit-learn`, `XGBoost`, `LightGBM`
- **API & Serving**: `FastAPI`, `Uvicorn`, `Pydantic`
- **Serialization & Validation**: `Pandera` (schema enforcement), `YAML` (configuration)
- **Tests**: `Pytest`
- **Containerization**: `Docker`, `Docker Compose`
- **Monitoring & Observability**: `Prometheus`, `Grafana`, `Evidently AI`
- **Orchestration & Tooling**: `MLflow` (tracking), `UV` (dependency management), `Make` (task automation)

---

## 2. Documentation

For deep dives into specific parts of the project, refer to the following documentation:

- 🏗️ **[Data Pipeline](docs/data_pipeline.md)**: Details on the Medallion architecture (Bronze, Silver, Gold), cleaning rules, and data drift monitoring.
- 🧠 **[Model Training](docs/model_training.md)**: Information on preprocessing, the Model Factory, and the MLflow lifecycle.

---

## 3. Project Structure

```text
├── app/                  # FastAPI Application
│   ├── services/         # Business logic (ModelProvider, FeatureService)
│   ├── routes.py         # API endpoints
│   └── main.py           # App entry point & service injection
├── config/               # Model and pipeline YAML configurations
├── data/                 # Data lake (Landing, Bronze, Silver, Gold zones)
├── docs/                 # Detailed documentation (Data Pipeline, Training)
├── grafana/              # Grafana dashboards and provisioning
├── models/               # Local storage for serialized models
├── notebooks/            # EDA and Prototyping
├── src/passos_magicos/   # Core logic
│   ├── core/             # Path management and global configs
│   ├── data/             # Data pipeline scripts (Bronze, Silver, Gold)
│   └── models/           # Training and evaluation logic
├── tests/                # Unit and Integration tests
├── Dockerfile            # API container specification
├── docker-compose.yml    # Multi-container orchestration
├── Makefile              # Automation shortcuts
└── pyproject.toml        # Dependency definitions
```

---

## 4. Deploy Instructions

### Quick Start (Local)

1. **Install uv**: Ensure you have [uv](https://github.com/astral-sh/uv) installed.
2. **Initialize Environment**:
   ```bash
   uv sync
   make setup
   ```
3. **Run Full Pipeline**: (Ingestion -> Processing -> Training)
   ```bash
   make train-presets
   ```

### Running with Docker

To spin up the entire ecosystem (API, MLflow, Prometheus, Grafana) in one go:
```bash
make docker-up
```
The API will be available at `http://localhost:8000`, the MLflow UI at `http://localhost:5000`, Prometheus at `http://localhost:9090`, and Grafana at `http://localhost:3000`.

### Model Promotion (Production)

The FastAPI application is configured to load the model using the **`production`** alias. After training a model, you **must** promote it in the MLflow UI for it to be used by the API:

1.  **Access MLflow**: Open `http://localhost:5000` in your browser.
2.  **Navigate to Models**: Click on the **Models** tab in the top navigation bar.
3.  **Select the Model**: Click on `passos_magicos_defasagem_v1`.
4.  **Assign Alias**:
    - Click on the **Version** number you wish to use (e.g., *Version 1*).
    - In the **Aliases** section, click the `+` button and type `production`.
5.  **Reload API**: The API loads the model during startup. However, you can update the model in memory without a restart:
    - Use the **`POST /model/reload`** endpoint (see API section below).
    - If you prefer a full reset, restart the container or the local process.

To stop the services:
```bash
make docker-down
```

---

## 5. API

Interact with the API once it's running. Documentation is available at `http://127.0.0.1:8000/docs`.

### Health Check
**Description**: Verifies the system status, including whether the ML model and student database are correctly loaded.
```bash
curl http://127.0.0.1:8000/
```

### Model Discovery
**Description**: Retrieves metadata about the currently active model in production, including its version, alias, and hyperparameters.
```bash
curl http://127.0.0.1:8000/model
```

### Trigger New Training
**Description**: Initiates the training pipeline as a background task. The model will be trained and registered in MLflow.
```bash
curl -X POST http://127.0.0.1:8000/train
```

### Prediction by Student RA
**Description**: Performs a real-time risk prediction for a student using their unique RA. Features are fetched from the **SQLite Feature Store**.
```bash
curl -X POST http://127.0.0.1:8000/predict/1
```

### Dynamic Model Reload
**Description**: Reloads the active model from the provider (MLflow) into memory without restarting the application.
```bash
curl -X POST http://127.0.0.1:8000/model/reload
```

---

## 6. ML Pipeline Steps

The pipeline follows a robust automated workflow:

1. **Data Ingestion**: Historical data is landed in the `data/00_landing` zone as Excel files.
2. **Bronze (Raw/Clean)**: Inconsistent column names are normalized, and basic type casting is performed.
3. **Silver (Trusted)**:
   - **Feature Engineering**: Calculates years in institution, academic indices (INDE, IPP), and encodes categorical features.
   - **Data Drift Monitoring**: New batches are compared against the historical baseline using **Evidently AI**. Reports are saved in `data/reports`.
4. **Gold (Aggregated)**: Data is prepared in Parquet format and synchronized with the **SQLite Feature Store** for low-latency inference.
5. **Model Training**:
   - Fetches the Gold dataset.
   - Trains models (Random Forest, XGBoost, or LightGBM) based on current configuration.
   - Logs metrics, parameters, and artifacts to **MLflow**.
6. **Model Registry & Serving**: The best model is registered and promoted to *Production* state for the FastAPI application to consume.

---

## 7. Additional Usage

These `make` commands provide additional shortcuts for development and maintenance:

### Simulation & QA
- **Simulate Data Drift**: Ingests 2025 data to trigger the drift detection report.
  ```bash
  make simulate_drift
  ```
- **Run Tests**: Executes the `pytest` suite.
  ```bash
  make test
  ```

### MLOps & Infrastructure
- **MLflow UI**: Start the tracking server locally.
  ```bash
  make ui
  ```

### Model Configuration (`config.yaml`)

The project uses a centralized configuration file at `config/config.yaml` to manage experiments.

Centralizing parameters in a YAML file promotes decoupling between the model's logic and its hyperparameters. This ensures experiment reproducibility, facilitates audit trails within MLflow, and provides a single source of truth for the entire training pipeline without requiring code changes.

You can modify these parameters to test different algorithms and tuning strategies:

- **MLOps & Tracking**:
    - `experiment_name`: Grouping of related runs in MLflow.
    - `run_name`: Specific identifier for the current training execution.
    - `registered_model_name`: The name used in the MLflow Model Registry (consumed by the API).
- **Data & Features**:
    - `target_col`: The name of the column to predict (default: `target_defasagem`).
    - `numerical`, `categorical`, `binary`: Lists of features to be included in the training pipeline.
- **Model Selection & Tuning**:
    - `model.type`: Choose between `"random_forest"`, `"xgboost"`, or `"lightgbm"`.
    - `model.params`: Hyperparameters specific to the chosen algorithm (e.g., `n_estimators`, `max_depth`, `learning_rate`).

**To test a new model**:
1. Open `config/config.yaml`.
2. Change `model.type` to your desired algorithm (e.g., `"xgboost"`).
3. Update `model.params` with the appropriate keys for that algorithm.
4. Run `make train` to log the new experiment to MLflow.

**Advanced Training**:
You can also run training using a specific configuration file without modifying the default one:
```bash
make train CONFIG=config/my_experiment.yaml
```

### Maintenance
- **Clean Environment**: Deletes local data lake files, models, and artifacts (use with caution).
  ```bash
  make clean
  ```

---

## 8. Future Implementations

To further mature the platform, the following points are planned for future development:

- **☁️ Cloud Ingestion & Storage**: Transition from local filesystem to **S3/GCS** for the Medallion architecture (Bronze/Silver/Gold) to ensure scalability.
- **🧪 Automated Hyperparameter Tuning**: Integrate **Optuna** into the `train.py` script to automatically search for the best configuration for each model type.
- **📊 Advanced Drift Monitoring**: Export **Evidently AI** metrics to **Prometheus** to enable real-time alerting in **Grafana** whenever data drift is detected.
- **🛡️ API Security**: Implement **JWT (JSON Web Token)** authentication to secure the inference and training endpoints.
- **🔄 Continuous Training (CT)**: Build a fully automated pipeline where drift detection or model performance decay triggers a new training run and registration in MLflow.

---

## 9. Authors

Developed for FIAP - Tech Challenge 4 (ML Engineering Postgraduate Program).

* Izabelly de Oliveira Menezes | [Github](https://github.com/izabellyomenezes)
* Larissa Diniz da Silva | [Github](https://github.com/Ldiniz737)
* Luis Fernando Torres | [Github](https://github.com/luuisotorres)
* Rafael dos Santos Callegari | [Github](https://github.com/rafaelcallegari)
* Renato Massamitsu Zama Inomata | [Github](https://github.com/renatoinomata)
---

## 10. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.