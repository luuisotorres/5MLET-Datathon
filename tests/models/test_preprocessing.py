import pandas as pd
import numpy as np
import pytest

# Imports for data cleaning functions
from passos_magicos.data.preprocessing import (
    clean_fase,
    clean_idade,
    clean_genero,
    clean_ra,
    clean_pedra,
    clean_inde,
    clean_instituicao,
)

# Imports for ML transformations
from passos_magicos.models.ml_preprocessing import create_target_class, clip_indicators

# --- Data Cleaning Tests ---

def test_clean_fase():
    """Tests Phase extraction, handling 'ALFA', numbers, and nulls."""
    assert clean_fase("ALFA") == 0
    assert clean_fase("Fase 2") == 2
    assert clean_fase("FASE 7 (BOLSISTA)") == 7
    assert np.isnan(clean_fase("Desconhecida"))
    assert np.isnan(clean_fase(pd.NA))

def test_clean_idade():
    """Tests the critical bug of Excel dates '1900-01-XX' and float conversions."""
    assert clean_idade("1900-01-15 00:00:00") == 15
    assert clean_idade("18.0") == 18
    assert clean_idade(20) == 20
    assert clean_idade("Dezoito") == "Dezoito"
    assert pd.isna(clean_idade(np.nan))

def test_clean_genero():
    """Tests standardization of gender names."""
    assert clean_genero("Menina") == "F"
    assert clean_genero("Feminino") == "F"
    assert clean_genero("Menino") == "M"
    assert clean_genero("Masculino") == "M"
    assert clean_genero("Prefere não dizer") is None
    assert np.isnan(clean_genero(np.nan))

def test_clean_ra():
    """Tests extraction of student IDs from strings."""
    assert clean_ra("RA-12345") == 12345
    assert clean_ra("12345") == 12345
    assert np.isnan(clean_ra("Sem RA"))
    assert np.isnan(clean_ra(pd.NA))

def test_clean_pedra():
    """Tests spelling correction and dropping 'INCLUIR' placeholder."""
    assert clean_pedra("Agata") == "Ágata"
    assert clean_pedra("Ametista") == "Ametista"
    assert clean_pedra("INCLUIR") is None
    assert clean_pedra(pd.NA) is None

def test_clean_inde():
    """Tests float conversion and 'INCLUIR' handling for INDE scores."""
    assert clean_inde("8.5") == 8.5
    assert clean_inde(9.0) == 9.0
    assert np.isnan(clean_inde("INCLUIR"))
    assert np.isnan(clean_inde(pd.NA))
    assert np.isnan(clean_inde("Texto Invalido"))

def test_clean_instituicao():
    """Tests mapping of various school types to the 3 main categories."""
    assert clean_instituicao("Escola Pública") == "Pública"
    assert clean_instituicao("Rede Decisão") == "Privada"
    assert clean_instituicao("Privada - Programa de Apadrinhamento") == "Bolsista"
    assert clean_instituicao("Privada *Parcerias com Bolsa 100%") == "Bolsista"
    assert clean_instituicao("Colégio Militar") == "Outros"
    assert clean_instituicao(pd.NA) == "Outros"

# --- ML Transformation Tests ---

def test_create_target_class():
    """Tests categorization of academic lag."""
    assert create_target_class(-3.0) == 0
    assert create_target_class(-2.0) == 0
    assert create_target_class(-1.0) == 1
    assert create_target_class(0.0) == 2
    assert create_target_class(1.5) == 2

def test_clip_indicators():
    """Tests clipping of indicator features."""
    data = {
        'id': [1, 2, 3],
        'indicador_A': [-1, 5, 12],
        'indicador_B': [0, 10, 10],
        'other_col': [-100, 100, 50]
    }
    df = pd.DataFrame(data)
    
    df_clipped = clip_indicators(df, min_val=0, max_val=10)
    
    assert df_clipped['indicador_A'].min() >= 0
    assert df_clipped['indicador_A'].max() <= 10
    assert df_clipped['indicador_A'].tolist() == [0, 5, 10]
    assert df_clipped['other_col'].tolist() == [-100, 100, 50]

def test_map_gender():
    """Tests manual mapping of gender."""
    df = pd.DataFrame({'genero': ['M', 'F', 'M', 'F']})
    # Import locally to avoid circular dependency if any, or just rely on top import if updated
    from passos_magicos.models.ml_preprocessing import map_gender
    
    df_mapped = map_gender(df)
    assert df_mapped['genero'].tolist() == [1, 0, 1, 0]
