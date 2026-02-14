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