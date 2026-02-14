import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from typing import Dict, List, Callable

from passos_magicos.data import FeatureNames as FN

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File Paths
BRONZE_FILE_PATH = Path("data/01_bronze/PEDE_2022-24.xlsx")
SILVER_DIR_PATH = Path("data/02_silver")

# Columns to keep in the Silver layer (standardized across years)
COLUMNS_TO_KEEP = [
    FN.RA, FN.ANO_DADOS, FN.FASE, FN.IDADE, FN.GENERO,
    FN.ANOS_NA_INSTITUICAO, FN.INSTITUICAO, FN.PEDRA_ATUAL,
    FN.INDE, FN.IAA, FN.IEG, FN.IPS, FN.IDA, FN.IPV,
    FN.IAN, FN.IPP, FN.DEFASAGEM
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


#  Data cleaning functions
def clean_fase(valor):
    if pd.isna(valor):
        return np.nan
    str_val = str(valor).upper().strip()
    if 'ALFA' in str_val:
        return 0
    match = re.search(r'(\d+)', str_val)
    if match:
        return int(match.group(1))
    return np.nan


def clean_idade(valor):
    if pd.isna(valor):
        return valor
    str_val = str(valor)
    if str_val.startswith('1900-01-'):
        try:
            return pd.to_datetime(str_val).day
        except (ValueError, TypeError):
            return np.nan
    try:
        return int(float(str_val))
    except (ValueError, TypeError):
        return valor


def clean_genero(valor):
    map_generos = {'Menina': 'F', 'Menino': 'M',
                   'Feminino': 'F', 'Masculino': 'M'}
    if pd.isna(valor):
        return np.nan
    return map_generos.get(valor)


def clean_ra(valor):
    if pd.isna(valor):
        return np.nan
    str_val = str(valor).upper().strip()
    match = re.search(r'(\d+)', str_val)
    if match:
        return int(match.group(1))
    return np.nan


def clean_pedra(valor):
    if pd.isna(valor) or valor == 'INCLUIR':
        return None
    if valor == 'Agata':
        return 'Ágata'
    return valor


def clean_inde(valor):
    if pd.isna(valor) or valor == 'INCLUIR':
        return np.nan
    try:
        return float(valor)
    except ValueError:
        return np.nan


def clean_instituicao(valor):
    map_instituicao = {
        'Escola Pública': 'Pública',
        'Pública': 'Pública',
        'Rede Decisão': 'Privada',
        'Escola JP II': 'Privada',
        'Privada': 'Privada',
        'Privada - Programa de Apadrinhamento': 'Bolsista',
        'Privada - Programa de apadrinhamento': 'Bolsista',
        'Privada *Parcerias com Bolsa 100%': 'Bolsista',
        'Privada - Pagamento por *Empresa Parceira': 'Bolsista',
        'Bolsista Universitário *Formado (a)': 'Bolsista',
        'Concluiu o 3º EM': 'Outros',
        'Nenhuma das opções acima': 'Outros',
    }
    if pd.isna(valor):
        return 'Outros'
    return map_instituicao.get(valor, 'Outros')


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
    return df


def _transform_2024(df: pd.DataFrame) -> pd.DataFrame:
    return df


STRATEGIES: Dict[int, Callable[[pd.DataFrame], pd.DataFrame]] = {
    2022: _transform_2022, 2023: _transform_2023, 2024: _transform_2024
}


# Core preprocessing pipeline
def process_sheet(xls: pd.ExcelFile, sheet_name: str, year: int, mapping: Dict[str, str], keep_columns: List[str]) -> pd.DataFrame:
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
                f"Column '{col}' is missing in year {year}. Creating as pd.NA.")
            df_clean[col] = pd.NA

    df_clean = df_clean[keep_columns]
    return df_clean


def main():
    logging.info("Starting Bronze to Silver extraction...")
    SILVER_DIR_PATH.mkdir(parents=True, exist_ok=True)

    try:
        xls = pd.ExcelFile(BRONZE_FILE_PATH)
    except FileNotFoundError:
        logging.error(f"Bronze file not found at {BRONZE_FILE_PATH}.")
        return

    sheets_to_process = {"PEDE2022": 2022, "PEDE2023": 2023, "PEDE2024": 2024}

    for sheet_name, year in sheets_to_process.items():
        mapping = COLUMN_MAPPINGS.get(year, {})
        df_silver = process_sheet(
            xls, sheet_name, year, mapping, COLUMNS_TO_KEEP)

        output_path = SILVER_DIR_PATH / f"alunos_{year}_clean.parquet"
        df_silver.to_parquet(output_path, index=False)
        logging.info(f"Successfully saved {output_path}")

    logging.info("Silver Layer Pipeline completed.")


if __name__ == "__main__":
    main()
