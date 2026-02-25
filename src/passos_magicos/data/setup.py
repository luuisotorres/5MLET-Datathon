import os
from passos_magicos.core.paths import ProjectPaths as PP


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

    for dir in directories:
        os.makedirs(dir, exist_ok=True)


if __name__ == "__main__":
    main()
    print("✅ All necessary directories have been created.")