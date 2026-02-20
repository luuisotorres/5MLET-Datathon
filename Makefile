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

train:
	@echo "==> Training model with current config..."
	uv run python -m src.passos_magicos.models.train

ui:
	@echo "==> Starting MLflow UI..."
	uv run mlflow ui

# The 'all' target runs the full pipeline from raw data to trained model
all: data train