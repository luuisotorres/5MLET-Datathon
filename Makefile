.PHONY: setup seed_baseline simulate_drift bronze silver gold data clean test

# ==============================================================================
# Data Pipeline Commands
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

data: clean seed_baseline bronze silver gold
	@echo "==> Data pipeline completed successfully!"

# ==============================================================================
# Drift Simulation Command (Presentation Mode)
# ==============================================================================
simulate_drift:
	@echo "==> Simulating 2025 data arrival..."
	uv run python -c "import shutil; shutil.copy('data/files/PEDE_2025.xlsx', 'data/00_landing/')"
	@echo "==> Running Bronze to extract new batch..."
	uv run src/passos_magicos/data/make_bronze.py
	@echo "==> Running Silver to trigger Stateful Drift Detection..."
	uv run src/passos_magicos/data/make_silver.py
	@echo "==> Simulation complete! Check data/reports for the Drift HTML."

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