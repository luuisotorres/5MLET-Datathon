.PHONY: silver gold data clean

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
# Utility Commands (Cross-Platform)
# ==============================================================================

clean:
	@echo "==> Triggering cleanup script..."
	uv run python src/passos_magicos/data/cleanup_environment.py