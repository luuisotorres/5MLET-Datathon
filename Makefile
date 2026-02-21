.PHONY: silver gold data clean test

# ==============================================================================
# Data Pipeline Commands
# ==============================================================================

silver:
	@echo "==> Running Bronze to Silver pipeline..."
	uv run python src/passos_magicos/data/make_silver.py

gold:
	@echo "==> Running Silver to Gold pipeline..."
	uv run python src/passos_magicos/data/make_gold.py

# The 'data' target runs 'silver' first, then 'gold'
data: silver gold
	@echo "==> Data pipeline completed successfully!"

# ==============================================================================
# Utility Commands
# ==============================================================================

clean:
	@echo "==> Triggering cleanup script..."
	uv run python src/passos_magicos/data/cleanup_environment.py

# ==============================================================================
# Quality Assurance Commands
# ==============================================================================

test:
	@echo "==> Running Pytest suite..."
	uv run pytest tests/ -v
	@echo "==> All tests passed successfully!"

# ==============================================================================
# Model Training & MLOps Commands
# ==============================================================================

CONFIG ?= config/config.yaml

train:
	@echo "==> Training model with config: $(CONFIG)..."
	uv run python -m src.passos_magicos.models.train --config $(CONFIG)

ui:
	@echo "==> Starting MLflow UI..."
	uv run mlflow ui

# The 'all' target runs the full pipeline from raw data to trained model
all: data train

# ==============================================================================
# API Commands
# ==============================================================================

# Define the port as a variable so you can change it easily if needed
PORT ?= 8000

.PHONY: run-api

run-api:
	@echo "==> Starting FastAPI server on port $(PORT)..."
	@echo "==> Swagger UI documentation will be available at http://127.0.0.1:$(PORT)/docs"
	uv run uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload