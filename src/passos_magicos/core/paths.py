from pathlib import Path


class ProjectPaths:
    """Centralizes all directory and file paths for the project."""

    # Base directories
    DATA_DIR = Path("data")

    # Medal layers
    LANDING_DIR = DATA_DIR / "00_landing"
    BRONZE_DIR = DATA_DIR / "01_bronze"
    SILVER_DIR = DATA_DIR / "02_silver"
    GOLD_DIR = DATA_DIR / "03_gold"
    ARCHIVE_DIR = DATA_DIR / "99_archive"
    FILES_DIR = DATA_DIR / "files"
    REPORTS_DIR = DATA_DIR / "reports"

    # Databases
    TRAINING_DATA_PARQUET_NAME = "training_data.parquet"
    OFFLINE_STORE_PARQUET_NAME = "feature_store.parquet"
    ONLINE_STORE_DB = Path("feature_store_online.db")
    MLFLOW_DB = Path("mlflow.db")
