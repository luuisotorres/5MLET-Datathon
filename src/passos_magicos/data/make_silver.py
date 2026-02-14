import pandas as pd
import logging
from typing import Dict, List, Callable

from passos_magicos.data import FeatureNames as FN
from passos_magicos.data import ProjectPaths as PP
from passos_magicos.data.preprocessing import (
    clean_fase, clean_genero,
    clean_idade, clean_inde,
    clean_pedra, clean_ra,
    clean_instituicao,
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Columns to keep in the Silver layer (standardized across years)
COLUMNS_TO_KEEP = [
    FN.RA,
    FN.ANO_DADOS,
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
]

COLUMN_MAPPINGS = {
    2022: {
        'RA': FN.RA,
        'Fase': FN.FASE,
        'Idade 22': FN.IDADE,
        'Gênero': FN.GENERO,
        'Ano ingresso': FN.ANO_INGRESSO,
        'Instituição de ensino': FN.INSTITUICAO,
        'Pedra 22': FN.PEDRA_ATUAL,
        'INDE 22': FN.INDE,
        'IAA': FN.IAA,
        'IEG': FN.IEG,
        'IPS': FN.IPS,
        'IDA': FN.IDA,
        'IPV': FN.IPV,
        'IAN': FN.IAN,
        'Defas': FN.DEFASAGEM,
    },
    2023: {
        'RA': FN.RA,
        'Fase': FN.FASE,
        'Idade': FN.IDADE,
        'Gênero': FN.GENERO,
        'Ano ingresso': FN.ANO_INGRESSO,
        'Instituição de ensino': FN.INSTITUICAO,
        'Pedra 2023': FN.PEDRA_ATUAL,
        'INDE 2023': FN.INDE,
        'IAA': FN.IAA,
        'IEG': FN.IEG,
        'IPS': FN.IPS,
        'IDA': FN.IDA,
        'IPV': FN.IPV,
        'IAN': FN.IAN,
        'IPP': FN.IPP,
        'Defasagem': FN.DEFASAGEM,
    },
    2024: {
        'RA': FN.RA,
        'Fase': FN.FASE,
        'Idade': FN.IDADE,
        'Gênero': FN.GENERO,
        'Ano ingresso': FN.ANO_INGRESSO,
        'Instituição de ensino': FN.INSTITUICAO,
        'Pedra 2024': FN.PEDRA_ATUAL,
        'INDE 2024': FN.INDE,
        'IAA': FN.IAA,
        'IEG': FN.IEG,
        'IPS': FN.IPS,
        'IDA': FN.IDA,
        'IPV': FN.IPV,
        'IAN': FN.IAN,
        'IPP': FN.IPP,
        'Defasagem': FN.DEFASAGEM,
    }
}


def _apply_global_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all the hygienic transformations to the standardized dataframe."""
    logging.info("Applying global hygiene and feature engineering...")

    # Apply individual column transformations
    if FN.IDADE in df.columns:
        df[FN.IDADE] = df[FN.IDADE].apply(clean_idade)
    if FN.GENERO in df.columns:
        df[FN.GENERO] = df[FN.GENERO].apply(clean_genero)
    if FN.FASE in df.columns:
        df[FN.FASE] = df[FN.FASE].apply(clean_fase)
    if FN.RA in df.columns:
        df[FN.RA] = df[FN.RA].apply(clean_ra)
    if FN.PEDRA_ATUAL in df.columns:
        df[FN.PEDRA_ATUAL] = df[FN.PEDRA_ATUAL].apply(clean_pedra)
    if FN.INDE in df.columns:
        df[FN.INDE] = df[FN.INDE].apply(clean_inde)
    if FN.INSTITUICAO in df.columns:
        df[FN.INSTITUICAO] = df[FN.INSTITUICAO].apply(clean_instituicao)

    # Feature Engineering: Anos na Instituição
    if FN.ANO_INGRESSO in df.columns and FN.ANO_DADOS in df.columns:
        # Convert to numeric just in case there are strings
        ano_ingresso_num = pd.to_numeric(df[FN.ANO_INGRESSO], errors='coerce')
        df[FN.ANOS_NA_INSTITUICAO] = df[FN.ANO_DADOS] - ano_ingresso_num

    return df


# Year specific strategies
def _transform_2022(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Applying 2022 specific rules (Inferring IPP)...")
    cols_to_numeric = [FN.INDE, FN.IAN, FN.IDA, FN.IEG, FN.IAA, FN.IPS, FN.IPV]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[FN.IPP] = (
        (df[FN.INDE] - (
            df[FN.IAN] * 0.1 + df[FN.IDA] * 0.2 + df[FN.IEG] * 0.2 +
            df[FN.IAA] * 0.1 + df[FN.IPS] * 0.1 + df[FN.IPV] * 0.2
        )) / 0.1
    ).round(3)
    return df


def _transform_2023(df: pd.DataFrame) -> pd.DataFrame:
    """No specific transformations needed for 2023, but we keep the function for consistency and future-proofing."""
    return df


def _transform_2024(df: pd.DataFrame) -> pd.DataFrame:
    """No specific transformations needed for 2024, but we keep the function for consistency and future-proofing."""
    return df


STRATEGIES: Dict[int, Callable[[pd.DataFrame], pd.DataFrame]] = {
    2022: _transform_2022,
    2023: _transform_2023,
    2024: _transform_2024,
}


# Core preprocessing pipeline
def process_sheet(
    xls: pd.ExcelFile,
    sheet_name: str,
    year: int,
    mapping: Dict[str, str],
    keep_columns: List[str]
) -> pd.DataFrame:
    logging.info(f"Processing sheet: {sheet_name} (Year: {year})")
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Rename columns to standard Portuguese terms
    df = df.rename(columns=mapping)
    df[FN.ANO_DADOS] = year

    # Apply Global Cleanings (The functions you provided!)
    df = _apply_global_cleaning(df)

    # Apply Year-Specific Strategy (e.g., creating IPP for 2022)
    strategy_func = STRATEGIES.get(year)
    if strategy_func:
        df = strategy_func(df)

    # Schema Enforcement
    available_columns = [col for col in keep_columns if col in df.columns]
    df_clean = df[available_columns].copy()

    for col in keep_columns:
        if col not in df_clean.columns:
            logging.warning(
                f"Column '{col}' is missing in year {year}. Creating as pd.NA."
            )
            df_clean[col] = pd.NA

    df_clean = df_clean[keep_columns]
    return df_clean


def main():
    logging.info("Starting Bronze to Silver extraction...")
    PP.SILVER_DIR.mkdir(parents=True, exist_ok=True)

    try:
        xls = pd.ExcelFile(PP.BRONZE_FILE)
    except FileNotFoundError:
        logging.error(f"Bronze file not found at {PP.BRONZE_FILE}.")
        return

    sheets_to_process = {
        "PEDE2022": 2022,
        "PEDE2023": 2023,
        "PEDE2024": 2024,
    }

    for sheet_name, year in sheets_to_process.items():
        mapping = COLUMN_MAPPINGS.get(year, {})
        df_silver = process_sheet(
            xls,
            sheet_name,
            year,
            mapping,
            COLUMNS_TO_KEEP,
        )

        output_path = PP.SILVER_DIR / f"alunos_{year}_clean.parquet"
        df_silver.to_parquet(output_path, index=False)
        logging.info(f"Successfully saved {output_path}")

    logging.info("Silver Layer Pipeline completed.")


if __name__ == "__main__":
    main()
