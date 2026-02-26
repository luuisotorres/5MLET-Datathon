import pandas as pd
import glob
import os
import shutil
import logging
from datetime import datetime

from passos_magicos.core.paths import ProjectPaths as PP

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    logging.info("Starting Bronze layer creation...")
    files_in_landing = glob.glob(os.path.join(PP.LANDING_DIR, "*"))
    if not files_in_landing:
        logging.warning("No files found in the landing directory.")
        return

    for file_path in files_in_landing:
        file_name = os.path.basename(file_path)
        logging.info(f"Processing file: {file_name}")
        try:
            with pd.ExcelFile(file_path) as excel:
                for sheet in excel.sheet_names:
                    logging.info(f"Processing sheet: {sheet}")
                    df = pd.read_excel(excel, sheet_name=sheet)
                    df["metadata_source"] = file_name
                    df["metadata_sheet"] = sheet
                    df["metadata_ingestion_date"] = datetime.now()
                    df = df.astype(str)

                    df.to_parquet(
                        f"{PP.BRONZE_DIR}/bronze_{sheet}.parquet", index=False
                    )

            # Moving file to archive after processing
            archive_path = os.path.join(PP.ARCHIVE_DIR, file_name)

            # Removes the file in the archive if it already exists to avoid errors when moving the new file
            if os.path.exists(archive_path):
                os.remove(archive_path)

            shutil.move(file_path, archive_path)
            logging.info(f"Successfully processed and archived: {file_name}")

        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}")

if __name__ == "__main__":
    main()