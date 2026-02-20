from pathlib import Path

class ProjectPaths:
    """Centralizes all directory and file paths for the project."""

    # Base directories
    DATA_DIR = Path("data")

    # Medal layers
    BRONZE_FILE = DATA_DIR / "01_bronze" / "PEDE_2022-24.xlsx"
    SILVER_DIR = DATA_DIR / "02_silver"
    GOLD_DIR = DATA_DIR / "03_gold"

    # Databases
    ONLINE_STORE_DB = Path("feature_store_online.db")
    MLFLOW_DB = Path("mlflow.db")
