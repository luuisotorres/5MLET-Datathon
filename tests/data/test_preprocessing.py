import pandas as pd
import numpy as np
import pytest

from passos_magicos.data.preprocessing import (
    clean_fase,
    clean_idade,
    clean_genero,
    clean_ra,
    clean_pedra,
    clean_inde,
    clean_instituicao,
)


def test_clean_fase():
    """Tests Phase extraction, handling 'ALFA', numbers, and nulls."""
    assert clean_fase("ALFA") == 0
    assert clean_fase("Fase 2") == 2
    assert clean_fase("FASE 7 (BOLSISTA)") == 7
    assert np.isnan(clean_fase("Desconhecida"))  # Should fail gracefully
    assert np.isnan(clean_fase(pd.NA))


def test_clean_idade():
    """Tests the critical bug of Excel dates '1900-01-XX' and float conversions."""
    # The Excel Date bug (1900-01-15 -> 15 years old)
    assert clean_idade("1900-01-15 00:00:00") == 15

    # Normal floats/strings
    assert clean_idade("18.0") == 18
    assert clean_idade(20) == 20

    # Graceful failure for weird text
    assert (
        clean_idade("Dezoito") == "Dezoito"
    )  # Assuming your code returns the original string here
    assert pd.isna(clean_idade(np.nan))


def test_clean_genero():
    """Tests standardization of gender names."""
    assert clean_genero("Menina") == "F"
    assert clean_genero("Feminino") == "F"
    assert clean_genero("Menino") == "M"
    assert clean_genero("Masculino") == "M"

    # Unknown values should return None/NaN based on dictionary .get()
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
    # Public Schools
    assert clean_instituicao("Escola Pública") == "Pública"

    # Private Schools
    assert clean_instituicao("Rede Decisão") == "Privada"

    # Scholarship (Bolsista)
    assert clean_instituicao("Privada - Programa de Apadrinhamento") == "Bolsista"
    assert clean_instituicao("Privada *Parcerias com Bolsa 100%") == "Bolsista"

    # Unknown / Null fallback
    assert clean_instituicao("Colégio Militar") == "Outros"
    assert clean_instituicao(pd.NA) == "Outros"
