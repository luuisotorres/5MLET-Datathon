import logging
from pathlib import Path

from passos_magicos.core.paths import ProjectPaths as PP


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_parquet_files(directory: Path):
    """Deletes all .parquet files in a given directory."""
    if not directory.exists():
        logging.warning(f"Directory {directory} does not exist. Skipping.")
        return

    count = 0
    for file_path in directory.glob("*.parquet"):
        try:
            file_path.unlink()
            count += 1
            logging.info(f"Deleted: {file_path.name}")
        except Exception as e:
            logging.error(f"Failed to delete {file_path.name}: {e}")

    logging.info(f"Cleared {count} files from {directory.name}")


def clean_database(db_path: Path):
    """Deletes the SQLite database file."""
    if db_path.exists():
        try:
            db_path.unlink()
            logging.info(f"Deleted database: {db_path.name}")
        except Exception as e:
            logging.error(f"Failed to delete database {db_path.name}: {e}")
    else:
        logging.info(f"Database {db_path.name} not found. Skipping.")


def main():
    logging.info("Starting data cleanup process...")

    clean_parquet_files(PP.LANDING_DIR)
    clean_parquet_files(PP.BRONZE_DIR)
    clean_parquet_files(PP.SILVER_DIR)
    clean_parquet_files(PP.GOLD_DIR)
    clean_parquet_files(PP.ARCHIVE_DIR)
    clean_database(PP.ONLINE_STORE_DB)

    logging.info("Cleanup completed! Environment is ready for a fresh pipeline run.")


if __name__ == "__main__":
    main()
