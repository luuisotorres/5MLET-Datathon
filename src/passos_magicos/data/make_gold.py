import pandas as pd
import sqlite3
import logging
from pathlib import Path

from passos_magicos.data import FeatureNames as FN
from passos_magicos.core.paths import ProjectPaths as PP

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_silver_data(silver_dir: Path) -> pd.DataFrame:
    """Loads and concatenates all Silver layer Parquet files."""
    logging.info(f"Loading Parquet files from {silver_dir}...")

    parquet_files = list(silver_dir.glob("silver_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError("No Parquet files found in the Silver layer.")

    dataframes = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        dataframes.append(df)
        logging.info(f"Loaded {file_path.name} ({len(df)} rows)")

    # Combine all years into a single DataFrame
    df_combined = pd.concat(dataframes, ignore_index=True)
    logging.info(f"Successfully concatenated all data. Total rows: {len(df_combined)}")

    return df_combined


def engineer_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts data temporally and creates the Target for Machine Learning."""
    logging.info("Sorting data chronologically by student (RA)...")

    # Sorting data to ensure .shift(-1) work as intended
    df = df.sort_values(by=[FN.RA, FN.ANO_DADOS]).reset_index(drop=True)

    logging.info("Creating Target variable (Defasagem of the next year)...")
    # Groups by RA, gets the 'Defasagem' of the next row (next year)
    df[FN.TARGET_DEFASAGEM] = df.groupby(FN.RA)[FN.DEFASAGEM].shift(-1)


    return df


def save_offline_store(df: pd.DataFrame, gold_dir: Path):
    """Saves the full historical data and the training subset as Parquet."""
    gold_dir.mkdir(parents=True, exist_ok=True)

    # Feature Store (Full History)
    feature_store_path = gold_dir / "feature_store.parquet"
    df.to_parquet(feature_store_path, index=False)
    logging.info(f"Offline Feature Store saved at {feature_store_path}")

    # Training Data (Only rows where the Target is NOT null)
    # The last year for every student will have a NaN target (since we don't know the future yet)
    df_train = df.dropna(subset=[FN.TARGET_DEFASAGEM]).copy()
    train_path = gold_dir / "train_data.parquet"
    df_train.to_parquet(train_path, index=False)
    logging.info(
        f"Training dataset saved at {train_path} ({len(df_train)} valid transitions)"
    )


def save_online_store(df: pd.DataFrame, db_path: Path):
    """Saves only the most recent snapshot of each student to a SQLite Database."""
    logging.info("Building Online Store (SQLite) for fast API inference...")

    # Keeping last is crucial because the last row for each RA will have the most recent year and features.
    df_latest = df.drop_duplicates(subset=[FN.RA], keep="last").copy()

    # Dropping the target column for the online store since inference shouldn't have access to the future
    if FN.TARGET_DEFASAGEM in df_latest.columns:
        df_latest = df_latest.drop(columns=[FN.TARGET_DEFASAGEM])

    try:
        # Connect to SQLite (creates the file if it doesn't exist)
        conn = sqlite3.connect(db_path)

        # Save dataframe as a SQL table, replacing the old one if it exists
        df_latest.to_sql("aluno_features", conn, if_exists="replace", index=False)
        conn.close()

        logging.info(
            f"Online Store successfully updated at {db_path} ({len(df_latest)} active students)"
        )
    except Exception as e:
        logging.error(f"Failed to update Online Store: {e}")


def main():
    logging.info("Starting Silver to Gold transformation...")

    try:
        # Extract
        df_silver = load_silver_data(PP.SILVER_DIR)

        # Transform (Time-series shifting)
        df_gold = engineer_features_and_target(df_silver)
        df_gold.drop(columns=[FN.METADATA_SHEET, FN.METADATA_SOURCE], inplace=True)

        # Load (Offline Batch)
        save_offline_store(df_gold, PP.GOLD_DIR)

        # Load (Online Real-time)
        save_online_store(df_gold, PP.ONLINE_STORE_DB)

        logging.info(
            "Gold Layer Pipeline completed successfully! Ready for Model Training."
        )

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
