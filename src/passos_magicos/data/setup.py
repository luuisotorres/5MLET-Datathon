import os
import logging
from passos_magicos.core.paths import ProjectPaths as PP

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Creates all necessary directories for the project."""

    directories = [
        PP.DATA_DIR,
        PP.LANDING_DIR,
        PP.BRONZE_DIR,
        PP.SILVER_DIR,
        PP.GOLD_DIR,
        PP.ARCHIVE_DIR,
        PP.FILES_DIR,
    ]

    logging.info("Creating directories")
    try:
        for dir in directories:
            os.makedirs(dir, exist_ok=True)
        logging.info("All necessary directories have been created.")

    except Exception as e:
        logging.error(f"Failed to create directories: {e}")


if __name__ == "__main__":
    main()
