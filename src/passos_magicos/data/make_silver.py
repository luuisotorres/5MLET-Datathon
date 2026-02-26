import os
import glob
import logging
import re
import pandas as pd
import pandera.pandas as pa
import unidecode

from evidently import Report
from evidently.presets import DataDriftPreset

from passos_magicos.core.paths import ProjectPaths as PP
from passos_magicos.data import FeatureNames as FN
from passos_magicos.data.preprocessing import (
    clean_fase,
    clean_genero,
    clean_idade,
    clean_inde,
    clean_pedra,
    clean_ra,
    clean_instituicao,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Columns strictly required for the Silver layer
COLUMNS_TO_KEEP = [
    FN.RA,
    FN.FASE,
    FN.IDADE,
    FN.GENERO,
    FN.ANOS_NA_INSTITUICAO,
    FN.INSTITUICAO,
    FN.PEDRA_ATUAL,
    FN.INDE,
    FN.IAA,
    FN.IEG,
    FN.IPS,
    FN.IDA,
    FN.IPV,
    FN.IAN,
    FN.IPP,
    FN.DEFASAGEM,
    FN.ANO_DADOS,
    FN.METADATA_SOURCE,
    FN.METADATA_SHEET,
]

# Business rules weights for IPP reconstruction
IPP_WEIGHTS = {
    "IAN": 0.1,
    "IDA": 0.2,
    "IEG": 0.2,
    "IAA": 0.1,
    "IPS": 0.1,
    "IPV": 0.2,
    "BASE_DIVISOR": 0.1,
}

# ==========================================
# 1. CORE TRANSFORMATIONS
# ==========================================


def _find_and_assign(df: pd.DataFrame, possible_names: list) -> pd.Series:
    """Searches for the first matching column in the priority list."""
    for name in possible_names:
        if name in df.columns:
            return df[name]
    # Binds to the dataframe's index to prevent Index Alignment ValueErrors
    return pd.Series(pd.NA, index=df.index, dtype=object)


def _extract_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target-driven extraction. Uses the metadata year to prioritize and fetch
    the correct column (e.g., inde_2023 vs inde_23 vs inde) while ignoring
    historical overlapping columns (e.g., pedra_20, pedra_21).
    """
    # 1. Standardize all incoming column names (lowercase, no spaces, no accents)
    df.columns = [
        unidecode.unidecode(str(c).strip().lower()).replace(" ", "_")
        for c in df.columns
    ]

    # 2. Identify the year from metadata (e.g., "PEDE2023" -> full: "2023", short: "23")
    metadata_sheet = str(df.get(FN.METADATA_SHEET, [""]).iloc[0])
    match = re.search(r"(\d{2})(\d{2})", metadata_sheet)
    if match:
        year_full = str(match.group(0))
        year_short = str(match.group(2))
    else:
        year_full, year_short = "", ""

    extracted = pd.DataFrame()

    # 3. Exact mappings
    extracted[FN.RA] = _find_and_assign(df, ["ra"])
    extracted[FN.FASE] = _find_and_assign(df, ["fase"])
    extracted[FN.GENERO] = _find_and_assign(df, ["genero"])
    extracted[FN.ANO_INGRESSO] = _find_and_assign(df, ["ano_ingresso"])
    extracted[FN.INSTITUICAO] = _find_and_assign(
        df, ["instituicao_de_ensino", "instituicao"]
    )
    extracted[FN.DEFASAGEM] = _find_and_assign(df, ["defasagem", "defas"])

    # 4. Temporal mappings with Priority Fallback
    extracted[FN.IDADE] = _find_and_assign(
        df, [f"idade_{year_full}", f"idade_{year_short}", "idade"]
    )
    extracted[FN.PEDRA_ATUAL] = _find_and_assign(
        df, [f"pedra_{year_full}", f"pedra_{year_short}", "pedra"]
    )
    extracted[FN.INDE] = _find_and_assign(
        df, [f"inde_{year_full}", f"inde_{year_short}", "inde"]
    )

    # 5. Indicators mapping (Buscando pelas siglas exatas do Excel)
    indicator_mapping = {
        FN.IAA: "iaa",
        FN.IEG: "ieg",
        FN.IPS: "ips",
        FN.IDA: "ida",
        FN.IPV: "ipv",
        FN.IAN: "ian",
        FN.IPP: "ipp",
    }

    for target_column, excel_acronym in indicator_mapping.items():
        extracted[target_column] = _find_and_assign(df, [excel_acronym])

    # 6. Preserve Lineage Metadata
    extracted[FN.METADATA_SHEET] = df.get(FN.METADATA_SHEET, pd.NA)
    extracted[FN.METADATA_SOURCE] = df.get(FN.METADATA_SOURCE, pd.NA)

    return extracted


def _calculate_missing_ipp(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates IPP for years where it is missing (like 2022) using the standard formula."""
    if df[FN.IPP].isna().all():
        logging.info(
            f"Column {FN.IPP} is missing/empty. Reconstructing via base formula..."
        )

        cols_to_numeric = [FN.INDE, FN.IAN, FN.IDA, FN.IEG, FN.IAA, FN.IPS, FN.IPV]
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df[FN.IPP] = (
            (
                df[FN.INDE]
                - (
                    df[FN.IAN] * IPP_WEIGHTS["IAN"]
                    + df[FN.IDA] * IPP_WEIGHTS["IDA"]
                    + df[FN.IEG] * IPP_WEIGHTS["IEG"]
                    + df[FN.IAA] * IPP_WEIGHTS["IAA"]
                    + df[FN.IPS] * IPP_WEIGHTS["IPS"]
                    + df[FN.IPV] * IPP_WEIGHTS["IPV"]
                )
            )
            / IPP_WEIGHTS["BASE_DIVISOR"]
        ).round(3)
    return df


def _apply_global_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Applies global hygiene and feature engineering rules."""
    if FN.IDADE in df.columns:
        df[FN.IDADE] = df[FN.IDADE].apply(clean_idade).astype(int)
    if FN.GENERO in df.columns:
        df[FN.GENERO] = df[FN.GENERO].apply(clean_genero).astype(str)
    if FN.FASE in df.columns:
        df[FN.FASE] = df[FN.FASE].apply(clean_fase).astype(int)
    if FN.RA in df.columns:
        df[FN.RA] = df[FN.RA].apply(clean_ra).astype(int)
    if FN.PEDRA_ATUAL in df.columns:
        df[FN.PEDRA_ATUAL] = df[FN.PEDRA_ATUAL].apply(clean_pedra)
    if FN.INDE in df.columns:
        df[FN.INDE] = df[FN.INDE].apply(clean_inde).astype(float)
    if FN.INSTITUICAO in df.columns:
        df[FN.INSTITUICAO] = df[FN.INSTITUICAO].apply(clean_instituicao).astype(str)
    if FN.IAA in df.columns:
        df[FN.IAA] = df[FN.IAA].astype(float)
    if FN.IEG in df.columns:
        df[FN.IEG] = df[FN.IEG].astype(float)
    if FN.IPS in df.columns:
        df[FN.IPS] = df[FN.IPS].astype(float)
    if FN.IDA in df.columns:
        df[FN.IDA] = df[FN.IDA].astype(float)
    if FN.IPV in df.columns:
        df[FN.IPV] = df[FN.IPV].astype(float)
    if FN.IAN in df.columns:
        df[FN.IAN] = df[FN.IAN].astype(float)
    if FN.IPP in df.columns:
        df[FN.IPP] = df[FN.IPP].astype(float)
    if FN.DEFASAGEM in df.columns:
        df[FN.DEFASAGEM] = df[FN.DEFASAGEM].astype(int)

    # Feature Engineering: Years in the Institution
    if FN.ANO_INGRESSO in df.columns:
        ano_ingresso_num = pd.to_numeric(df[FN.ANO_INGRESSO], errors="coerce")
        ano_dados_num = df[FN.METADATA_SHEET].str.extract(r"(\d{4})")[0].astype(int)
        df[FN.ANO_DADOS] = ano_dados_num
        df[FN.ANOS_NA_INSTITUICAO] = (ano_dados_num - ano_ingresso_num).astype(int)

    return df


# ==========================================
# 2. VALIDATION & MONITORING GATES
# ==========================================


def _get_historical_baseline() -> pd.DataFrame:
    """
    Reads all existing partitioned files in the Silver layer and concatenates
    them in memory to serve as the Reference Dataset for Data Drift monitoring.
    """
    silver_files = glob.glob(os.path.join(PP.SILVER_DIR, "silver_*.parquet"))
    if not silver_files:
        return pd.DataFrame()  # Returns empty if Cold Start

    dfs = [pd.read_parquet(f) for f in silver_files]
    return pd.concat(dfs, ignore_index=True)


def _validate_schema_pandera(df: pd.DataFrame) -> pd.DataFrame:
    """Enforces the Data Contract before persisting to Silver."""
    for col in COLUMNS_TO_KEEP:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[COLUMNS_TO_KEEP].copy()

    schema = pa.DataFrameSchema(
        {
            FN.METADATA_SHEET: pa.Column(str, coerce=True),
            FN.METADATA_SOURCE: pa.Column(str, coerce=True),
            FN.RA: pa.Column(nullable=True),
        }
    )

    return schema.validate(df)


def _monitor_data_drift(
    df_current: pd.DataFrame, batch_name: str, df_reference: pd.DataFrame
):
    """Triggers Evidently AI comparing the new batch against the historical baseline."""
    logging.info(f"Checking Data Drift for batch: {batch_name}...")

    features_to_monitor = [
        FN.IDADE,
        FN.FASE,
        FN.GENERO,
        FN.PEDRA_ATUAL,
        FN.DEFASAGEM,
        FN.INDE,
        FN.IAA,
        FN.IEG,
        FN.IPS,
        FN.IDA,
        FN.IPV,
        FN.IAN,
        FN.IPP,
    ]

    # 1. Filter out metadata, institution names, and RA
    df_ref_monitor = df_reference[features_to_monitor].copy()
    df_curr_monitor = df_current[features_to_monitor].copy()

    # 2. Initialize the report
    drift_report = Report([DataDriftPreset()], include_tests=True)

    # 3. Run the report on the clean, sliced DataFrames
    eval_result = drift_report.run(
        reference_data=df_ref_monitor,
        current_data=df_curr_monitor,
    )

    # 4. Save the HTML
    os.makedirs(PP.REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(PP.REPORTS_DIR, f"drift_report_{batch_name}.html")
    eval_result.save_html(report_path)

    # 5. Extract results
    results = eval_result.dict()

    try:
        tests = results["tests"]
        failed_tests = [test for test in tests if test["status"] == "FAIL"]
        if len(failed_tests) > 0:
            logging.warning(f"Data Drift detected! Check the report at: {report_path}")
        else:
            logging.info("Data distribution is stable. No drift detected.")

    except KeyError:
        logging.info(f"Report generated successfully at {report_path}")


# ==========================================
# 3. PIPELINE ORCHESTRATION
# ==========================================


def main():
    logging.info("Starting Bronze to Silver extraction pipeline...")

    bronze_files = glob.glob(os.path.join(PP.BRONZE_DIR, "*.parquet"))
    if not bronze_files:
        logging.info("No files found in the Bronze layer. Pipeline is idle.")
        return

    # 1. Fetch historical baseline in memory (if any exists)
    df_historical_reference = _get_historical_baseline()
    is_cold_start = df_historical_reference.empty

    if is_cold_start:
        logging.info("COLD START DETECTED: Silver layer is empty. Building baseline.")
    else:
        logging.info("INCREMENTAL LOAD DETECTED: Existing Silver data found.")

    for file_path in bronze_files:
        file_name = os.path.basename(file_path)

        # Extract year from filename (e.g., "bronze_PEDE2022.parquet" -> "2022")
        match = re.search(r"\d{4}", file_name)
        batch_year = match.group(0) if match else "unknown"

        expected_silver_path = os.path.join(
            PP.SILVER_DIR, f"silver_{batch_year}.parquet"
        )

        if os.path.exists(expected_silver_path):
            logging.info(
                f"SKIPPING: {file_name} (Silver target '{batch_year}' already exists)."
            )
            continue  # Skips the rest of the loop and goes to the next file!
        # ----------------------------------------------------------------

        logging.info(f"Processing: {file_name}")

        # Only loads into memory if it actually needs to be processed
        df = pd.read_parquet(file_path)

        # Line of assembly
        df = _extract_target_columns(df)
        df = _calculate_missing_ipp(df)
        df = _apply_global_cleaning(df)

        try:
            df_validated = _validate_schema_pandera(df)

            # 2. Stateful Processing (Drift Check)
            if is_cold_start:
                logging.info(
                    f"Skipping drift detection for {batch_year} (building baseline)."
                )
            else:
                logging.info(f"Validating {batch_year} against history...")
                _monitor_data_drift(df_validated, batch_year, df_historical_reference)

            # 3. Save as a partitioned file
            df_validated.to_parquet(expected_silver_path, index=False)
            logging.info(f"Saved partitioned file: {expected_silver_path}")

            # 4. Update the reference dataframe in memory for subsequent files in this batch
            if df_historical_reference.empty:
                df_historical_reference = df_validated
            else:
                if batch_year not in str(
                    df_historical_reference.get(FN.METADATA_SHEET, "")
                ):
                    df_historical_reference = pd.concat(
                        [df_historical_reference, df_validated], ignore_index=True
                    )

        except pa.errors.SchemaError as e:
            logging.error(f"Schema Validation Error in {file_name}: {e}")
            raise e

    logging.info("Bronze to Silver pipeline completed successfully.")


if __name__ == "__main__":
    main()
