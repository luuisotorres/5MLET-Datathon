import pandas as pd
import pytest
from pathlib import Path

from passos_magicos.data.constants import FeatureNames as FN
from passos_magicos.data.make_gold import (
    engineer_features_and_target,
    save_online_store,
)


@pytest.fixture
def mock_silver_history():
    """
    Simulates the concatenated Silver data with two students across multiple years.
    Student 1 has full history (22, 23, 24).
    Student 2 joined late (23, 24).
    """
    return pd.DataFrame(
        {
            FN.RA: ["RA-1", "RA-1", "RA-1", "RA-2", "RA-2"],
            FN.ANO_DADOS: [2022, 2023, 2024, 2023, 2024],
            FN.DEFASAGEM: [
                0,
                -1,
                -2,
                1,
                0,
            ],  # Student 1 gets worse over time, Student 2 gets worse too (1 -> 0)
            FN.INDE: [7.0, 6.5, 5.0, 8.0, 7.5],
        }
    )


def test_engineer_features_and_target_shifting(mock_silver_history):
    """Tests if the target creation (next year's defasagem) is shifted correctly per student."""

    df_gold = engineer_features_and_target(mock_silver_history)

    # Check if the target column was created
    assert FN.TARGET_DEFASAGEM in df_gold.columns

    # Check Student 1 (RA-1)
    ra1_data = df_gold[df_gold[FN.RA] == "RA-1"].sort_values(by=FN.ANO_DADOS)
    targets_ra1 = ra1_data[FN.TARGET_DEFASAGEM].tolist()

    # 2022 should target 2023's defasagem (-1)
    # 2023 should target 2024's defasagem (-2)
    # 2024 should be NaN (we don't know 2025 yet)
    assert targets_ra1[0] == -1.0
    assert targets_ra1[1] == -2.0
    assert pd.isna(targets_ra1[2])

    # Check Student 2 (RA-2) to ensure no data leaked from RA-1
    ra2_data = df_gold[df_gold[FN.RA] == "RA-2"].sort_values(by=FN.ANO_DADOS)
    targets_ra2 = ra2_data[FN.TARGET_DEFASAGEM].tolist()

    # 2023 should target 2024's defasagem (0)
    # 2024 should be NaN
    assert targets_ra2[0] == 0.0
    assert pd.isna(targets_ra2[1])


def test_online_store_drops_target_and_keeps_latest(mock_silver_history, tmp_path):
    """Tests if the SQLite preparation correctly isolates the latest snapshot and hides the future."""

    df_gold = engineer_features_and_target(mock_silver_history)

    # Use pytest's tmp_path fixture to create a temporary SQLite database
    db_path = tmp_path / "test_online_store.db"

    save_online_store(df_gold, db_path)

    import sqlite3

    conn = sqlite3.connect(db_path)
    df_saved = pd.read_sql("SELECT * FROM aluno_features", conn)
    conn.close()

    # Target column must NOT be in the online store
    assert FN.TARGET_DEFASAGEM not in df_saved.columns

    # Only the latest records (2024 for both) should be kept
    assert len(df_saved) == 2
    assert df_saved[df_saved[FN.RA] == "RA-1"][FN.ANO_DADOS].values[0] == 2024
    assert df_saved[df_saved[FN.RA] == "RA-2"][FN.ANO_DADOS].values[0] == 2024
