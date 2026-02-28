import logging
from pathlib import Path

from passos_magicos.core.paths import ProjectPaths as PP


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_files(directory: Path, extension: str = ".parquet"):
    """Deletes all files with the specified extension in a given directory."""
    if not directory.exists():
        logging.warning(f"Directory {directory} does not exist. Skipping.")
        return

    # Ensure extension starts with a dot
    if not extension.startswith("."):
        extension = f".{extension}"

    count = 0
    for file_path in directory.glob(f"*{extension}"):
        try:
            file_path.unlink()
            count += 1
            logging.info(f"Deleted: {file_path.name}")
        except Exception as e:
            logging.error(f"Failed to delete {file_path.name}: {e}")

    logging.info(f"Cleared {count} {extension} files from {directory.name}")


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

    # Clean parquet files from all data directories
    data_dirs = [
        PP.LANDING_DIR,
        PP.BRONZE_DIR,
        PP.SILVER_DIR,
        PP.GOLD_DIR,
        PP.ARCHIVE_DIR,
    ]

    for directory in data_dirs:
        clean_files(directory, extension=".parquet")

    clean_files(PP.ARCHIVE_DIR, extension=".xlsx")

    clean_database(PP.ONLINE_STORE_DB)

    logging.info("Cleanup completed! Environment is ready for a fresh pipeline run.")


if __name__ == "__main__":
    main()
