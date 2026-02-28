# ==============================================================================
# Configuration
# ==============================================================================
CONFIG ?= config/config.yaml
PORT ?= 8000

.DEFAULT_GOAL := help

.PHONY: help setup seed_baseline bronze silver gold data clean test simulate_drift train ui run-api docker-up docker-down docker-restart docker-logs docker-build train-presets

# ==============================================================================
# Help / Information
# ==============================================================================
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Infrastructure / Docker:"
	@echo "  docker-up        Start all services in Docker (API, MLflow, Prometheus, Grafana)"
	@echo "  docker-down      Stop and remove Docker containers"
	@echo "  docker-restart   Restart Docker services"
	@echo "  docker-logs      View Docker logs"
	@echo "  docker-build     Rebuild Docker images"
	@echo ""
	@echo "Data Pipeline & Training:"
	@echo "  data            Run the full data pipeline (clean -> gold)"
	@echo "  train           Train model using CONFIG (default: config/config.yaml)"
	@echo "  train-presets   Run full pipeline: data + baseline + train (LightGBM & XGBoost)"
	@echo ""
	@echo "Application & MLOps Services (Local):"
	@echo "  run-api         Start FastAPI server locally on PORT (default: 8000)"
	@echo "  ui              Start MLflow UI"
	@echo ""
	@echo "Fine-grained Data Pipeline:"
	@echo "  seed_baseline   Ingest historical baseline data (2022-2024)"
	@echo "  bronze          Run Landing to Bronze pipeline"
	@echo "  silver          Run Bronze to Silver pipeline"
	@echo "  gold            Run Silver to Gold pipeline"
	@echo ""
	@echo "Simulation & QA:"
	@echo "  simulate_drift  Simulate 2025 data arrival and trigger drift detection"
	@echo "  test             Run pytest suite"
	@echo "  setup           Initialize project folders"
	@echo "  clean           Cleanup environment (delete generated data/models)"

# ==============================================================================
# Infrastructure / Docker
# ==============================================================================
docker-up:
	@echo "==> Starting containers..."
	docker compose up -d
	@echo "==> Services available at:"
	@echo "    - API:        http://localhost:8000"
	@echo "    - MLflow:     http://localhost:5000"
	@echo "    - Prometheus: http://localhost:9090"
	@echo "    - Grafana:    http://localhost:3000"

docker-down:
	@echo "==> Stopping containers..."
	docker compose down

docker-restart: docker-down docker-up

docker-logs:
	docker compose logs -f

docker-build:
	@echo "==> Rebuilding images..."
	docker compose build

# ==============================================================================
# Model Training & Pipeline
# ==============================================================================
data: clean seed_baseline bronze silver gold
	@echo "==> Data pipeline completed successfully!"

train:
	@echo "==> Training model with config: $(CONFIG)..."
	uv run python -m src.passos_magicos.models.train --config $(CONFIG)

train-presets: data
	@echo "==> Training LightGBM model..."
	$(MAKE) train CONFIG=config/lightgbm.yaml
	@echo "==> Training XGBoost model..."
	$(MAKE) train CONFIG=config/xgboost.yaml

# ==============================================================================
# Application & MLOps Services
# ==============================================================================
run-api:
	@echo "==> Starting FastAPI server on port $(PORT)..."
	@echo "==> Swagger UI documentation will be available at http://127.0.0.1:$(PORT)/docs"
	uv run uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

ui:
	@echo "==> Starting MLflow UI..."
	uv run mlflow ui

# ==============================================================================
# Fine-grained Data Pipeline
# ==============================================================================
setup:
	@echo "==> Initializing project folders..."
	uv run src/passos_magicos/data/setup.py

seed_baseline: setup
	@echo "==> Ingesting historical baseline data (2022-2024)..."
	uv run python -c "import shutil; shutil.copy('data/files/PEDE_2022-24.xlsx', 'data/00_landing/')"
	@echo "==> Baseline data successfully placed in Landing zone."

bronze:
	@echo "==> Running Landing to Bronze pipeline..."
	uv run src/passos_magicos/data/make_bronze.py

silver:
	@echo "==> Running Bronze to Silver pipeline..."
	uv run src/passos_magicos/data/make_silver.py

gold:
	@echo "==> Running Silver to Gold pipeline..."
	uv run src/passos_magicos/data/make_gold.py

# ==============================================================================
# Simulation & Maintenance
# ==============================================================================
simulate_drift:
	@echo "==> Simulating 2025 data arrival..."
	uv run python -c "import shutil; shutil.copy('data/files/PEDE_2025.xlsx', 'data/00_landing/')"
	@echo "==> Running Bronze to extract new batch..."
	uv run src/passos_magicos/data/make_bronze.py
	@echo "==> Running Silver to trigger Stateful Drift Detection..."
	uv run src/passos_magicos/data/make_silver.py
	@echo "==> Simulation complete! Check data/reports for the Drift HTML."

test:
	@echo "==> Running Pytest suite..."
	uv run pytest tests/ -v
	@echo "==> All tests passed successfully!"

clean:
	@echo "==> Triggering cleanup script..."
	uv run python src/passos_magicos/data/cleanup_environment.py

