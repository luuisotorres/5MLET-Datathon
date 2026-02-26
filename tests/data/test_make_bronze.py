import os
import pytest
import pandas as pd
from unittest.mock import patch

from passos_magicos.data.make_bronze import main

# ==============================================================================
# FIXTURES (Mock File System Setup)
# ==============================================================================


@pytest.fixture
def mock_directories(tmp_path):
    """
    Creates temporary directories for testing and patches the ProjectPaths
    so the script uses these temporary folders instead of the real ones.
    """
    # Create isolated temp directories
    landing_dir = tmp_path / "00_landing"
    bronze_dir = tmp_path / "01_bronze"
    archive_dir = tmp_path / "archive"

    landing_dir.mkdir()
    bronze_dir.mkdir()
    archive_dir.mkdir()

    # Patch the ProjectPaths in the target module
    with (
        patch("passos_magicos.data.make_bronze.PP.LANDING_DIR", str(landing_dir)),
        patch("passos_magicos.data.make_bronze.PP.BRONZE_DIR", str(bronze_dir)),
        patch("passos_magicos.data.make_bronze.PP.ARCHIVE_DIR", str(archive_dir)),
    ):
        yield {
            "landing": str(landing_dir),
            "bronze": str(bronze_dir),
            "archive": str(archive_dir),
        }


@pytest.fixture
def sample_excel_file(mock_directories):
    """Creates a real, multi-sheet Excel file inside the fake Landing zone."""
    file_path = os.path.join(mock_directories["landing"], "dummy_data.xlsx")

    # Create two dataframes to simulate two tabs in the Excel file
    df_2023 = pd.DataFrame({"RA": [1, 2], "Nota": [8.5, 9.0]})
    df_2024 = pd.DataFrame({"RA": [3, 4], "Nota": [7.0, 10.0]})

    # Write them to the temp Excel file
    with pd.ExcelWriter(file_path) as writer:
        df_2023.to_excel(writer, sheet_name="PEDE2023", index=False)
        df_2024.to_excel(writer, sheet_name="PEDE2024", index=False)

    return file_path


# ==============================================================================
# UNIT TESTS
# ==============================================================================


def test_main_empty_landing(mock_directories, caplog):
    """Test if the script gracefully handles an empty landing zone."""
    # Execute the pipeline
    main()

    # Check if the correct warning was logged
    assert "No files found in the landing directory" in caplog.text

    # Ensure nothing was created in bronze
    bronze_files = os.listdir(mock_directories["bronze"])
    assert len(bronze_files) == 0


def test_main_successful_processing(mock_directories, sample_excel_file):
    """Test full E2E flow: Reading Excel, adding metadata, saving Parquet, moving to Archive."""

    # Execute the pipeline
    main()

    # 1. Assert file was moved out of Landing
    assert len(os.listdir(mock_directories["landing"])) == 0

    # 2. Assert file was moved into Archive
    archived_files = os.listdir(mock_directories["archive"])
    assert len(archived_files) == 1
    assert archived_files[0] == "dummy_data.xlsx"

    # 3. Assert Parquet files were created in Bronze (one per sheet)
    bronze_files = os.listdir(mock_directories["bronze"])
    assert len(bronze_files) == 2
    assert "bronze_PEDE2023.parquet" in bronze_files
    assert "bronze_PEDE2024.parquet" in bronze_files

    # 4. Assert Data integrity and Metadata Injection
    df_bronze = pd.read_parquet(
        os.path.join(mock_directories["bronze"], "bronze_PEDE2023.parquet")
    )

    # Check if metadata columns exist
    assert "metadata_source" in df_bronze.columns
    assert "metadata_sheet" in df_bronze.columns
    assert "metadata_ingestion_date" in df_bronze.columns

    # Check if metadata values are correct
    assert df_bronze["metadata_source"].iloc[0] == "dummy_data.xlsx"
    assert df_bronze["metadata_sheet"].iloc[0] == "PEDE2023"

    # Check if EVERYTHING was cast to string as requested in the script (df.astype(str))
    # Pandas represents strings as 'object' dtype
    assert df_bronze["RA"].dtype == "object"
    assert df_bronze["Nota"].dtype == "object"
    assert df_bronze["RA"].iloc[0] == "1"  # Should be a string '1', not an int 1


def test_main_archive_overwrite_handling(mock_directories, sample_excel_file):
    """Test if the script successfully overwrites an existing file in the archive."""

    # Pretend an older version of 'dummy_data.xlsx' already exists in the archive
    archive_path = os.path.join(mock_directories["archive"], "dummy_data.xlsx")
    with open(archive_path, "w") as f:
        f.write("I am an old file")

    # Execute the pipeline
    main()

    # If the overwrite logic (os.remove) fails, shutil.move will throw a FileExistsError
    # The fact that we reach this assertion means the pipeline handled it gracefully
    archived_files = os.listdir(mock_directories["archive"])
    assert len(archived_files) == 1

    # The new file should have replaced the text file
    assert pd.read_excel(archive_path, sheet_name="PEDE2023").shape[0] == 2


def test_main_error_handling(mock_directories, caplog):
    """Test if the script safely catches errors (like corrupted files) without crashing the loop."""

    # Create a corrupted/invalid Excel file (a simple text file disguised as .xlsx)
    bad_file_path = os.path.join(mock_directories["landing"], "corrupted.xlsx")
    with open(bad_file_path, "w") as f:
        f.write("This is not a real excel file.")

    main()

    # Check if the Exception was caught and logged
    assert "Error processing file corrupted.xlsx" in caplog.text

    # The file should still be in the landing zone (because it crashed before shutil.move)
    assert len(os.listdir(mock_directories["landing"])) == 1
