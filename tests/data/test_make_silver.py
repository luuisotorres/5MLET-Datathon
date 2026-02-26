import pytest
import pandas as pd
import numpy as np

from passos_magicos.data.make_silver import (
    _find_and_assign,
    _extract_target_columns,
    _calculate_missing_ipp,
    _apply_global_cleaning,
    _validate_schema_pandera,
    COLUMNS_TO_KEEP,
)
from passos_magicos.data import FeatureNames as FN

# ==============================================================================
# FIXTURES (Mock Data Setup)
# ==============================================================================


@pytest.fixture
def sample_raw_dataframe():
    """Mock dataframe representing raw data coming from the Bronze layer."""
    return pd.DataFrame(
        {
            "  RA ": ["RA-1234", "RA-5678"],
            "Fase ": ["ALFA", "FASE 8"],
            " Idade 23 ": ["10", "1900-01-21"],  # Simulating Excel bug
            "Gênero": ["Menina", "Masculino"],
            "Ano ingresso": ["2021", "2022"],
            "Instituição de ensino": ["Escola Pública", "Privada"],
            "Pedra 23": ["Ametista", "Topázio"],
            "INDE 2023": ["8.5", "7.2"],
            "iaa": ["9.0", "8.0"],
            "ieg": ["10.0", "7.5"],
            "ips": ["8.0", "7.0"],
            "ida": ["9.5", "8.5"],
            "ipv": ["7.0", "6.5"],
            "ian": ["8.0", "9.0"],
            "ipp": [pd.NA, pd.NA],  # Deliberately missing to test calculation
            "Defasagem": ["0", "1"],
            FN.METADATA_SHEET: ["PEDE2023", "PEDE2023"],
            FN.METADATA_SOURCE: ["arquivo_2023.xlsx", "arquivo_2023.xlsx"],
        }
    )


@pytest.fixture
def sample_indicators_dataframe():
    """Mock dataframe specifically for testing the IPP math formula."""
    return pd.DataFrame(
        {
            FN.INDE: [8.0],
            FN.IAN: [7.0],
            FN.IDA: [8.0],
            FN.IEG: [9.0],
            FN.IAA: [8.0],
            FN.IPS: [7.0],
            FN.IPV: [8.0],
            FN.IPP: [pd.NA],  # Should trigger calculation
        }
    )


# ==============================================================================
# UNIT TESTS
# ==============================================================================


def test_find_and_assign_found():
    """Test if it correctly finds a column from a list of possibilities."""
    df = pd.DataFrame({"col_b": [1, 2], "col_c": [3, 4]})

    # "col_a" is missing, but "col_b" exists. It should return "col_b"
    result = _find_and_assign(df, ["col_a", "col_b", "col_c"])

    pd.testing.assert_series_equal(result, df["col_b"])


def test_find_and_assign_not_found():
    """Test if it safely returns a series of pd.NA when no columns match."""
    df = pd.DataFrame(index=[0, 1])

    result = _find_and_assign(df, ["missing_col"])

    assert len(result) == 2
    assert pd.isna(result).all()


def test_extract_target_columns(sample_raw_dataframe):
    """Test if regex and mapping correctly normalize messy Excel columns."""
    df_extracted = _extract_target_columns(sample_raw_dataframe)

    # 1. Check if core columns were extracted and renamed properly
    assert FN.RA in df_extracted.columns
    assert FN.FASE in df_extracted.columns
    assert FN.IDADE in df_extracted.columns
    assert FN.INDE in df_extracted.columns

    # 2. Check if year-specific regex mapping worked (e.g., "Idade 23" -> FN.IDADE)
    # The first row for Idade 23 was "10"
    assert df_extracted[FN.IDADE].iloc[0] == "10"

    # 3. Check if metadata was preserved
    assert df_extracted[FN.METADATA_SHEET].iloc[0] == "PEDE2023"


def test_calculate_missing_ipp(sample_indicators_dataframe):
    """Test if the IPP formula calculates the correct mathematical output."""
    df_calc = _calculate_missing_ipp(sample_indicators_dataframe)

    # Expected Math:
    # Formula = (INDE - (IAN*0.1 + IDA*0.2 + IEG*0.2 + IAA*0.1 + IPS*0.1 + IPV*0.2)) / 0.1
    # Weighted sum = (7.0*0.1) + (8.0*0.2) + (9.0*0.2) + (8.0*0.1) + (7.0*0.1) + (8.0*0.2)
    # Weighted sum = 0.7 + 1.6 + 1.8 + 0.8 + 0.7 + 1.6 = 7.2
    # IPP = (8.0 - 7.2) / 0.1 = 0.8 / 0.1 = 8.0

    assert df_calc[FN.IPP].iloc[0] == 8.0


def test_calculate_missing_ipp_skips_if_present():
    """Test if the function leaves existing IPP values alone."""
    df = pd.DataFrame(
        {
            FN.INDE: [8.0],
            FN.IAN: [7.0],
            FN.IDA: [8.0],
            FN.IEG: [9.0],
            FN.IAA: [8.0],
            FN.IPS: [7.0],
            FN.IPV: [8.0],
            FN.IPP: [9.99],  # Already has data
        }
    )

    df_calc = _calculate_missing_ipp(df)
    assert df_calc[FN.IPP].iloc[0] == 9.99  # Should not be overwritten to 8.0


def test_apply_global_cleaning_feature_engineering():
    """Test if 'anos_na_instituicao' and 'ano_dados' are engineered correctly."""
    df = pd.DataFrame(
        {FN.ANO_INGRESSO: ["2021", "2018"], FN.METADATA_SHEET: ["PEDE2023", "PEDE2023"]}
    )

    df_clean = _apply_global_cleaning(df)

    # 2023 - 2021 = 2 years
    assert df_clean[FN.ANOS_NA_INSTITUICAO].iloc[0] == 2
    # 2023 - 2018 = 5 years
    assert df_clean[FN.ANOS_NA_INSTITUICAO].iloc[1] == 5
    # Extracted year
    assert df_clean[FN.ANO_DADOS].iloc[0] == 2023


def test_validate_schema_pandera():
    """Test if Pandera successfully enforces the schema and injects missing columns."""
    # Create a DF missing almost everything except RA and metadata
    df = pd.DataFrame(
        {
            FN.RA: ["12345"],
            FN.METADATA_SHEET: ["PEDE2023"],
            FN.METADATA_SOURCE: ["file.xlsx"],
        }
    )

    df_validated = _validate_schema_pandera(df)

    # 1. Check if all mandatory columns were successfully injected
    for col in COLUMNS_TO_KEEP:
        assert col in df_validated.columns

    # 2. Check if the missing columns were filled with NA
    assert pd.isna(df_validated[FN.IDADE].iloc[0])

    # 3. Check if existing data was preserved
    assert df_validated[FN.RA].iloc[0] == "12345"
