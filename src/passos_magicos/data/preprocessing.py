import pandas as pd
import numpy as np
import re


#  Data cleaning functions
def clean_fase(value):
    if pd.isna(value):
        return np.nan
    str_val = str(value).upper().strip()
    if "ALFA" in str_val:
        return 0
    match = re.search(r"(\d+)", str_val)
    if match:
        return int(match.group(1))
    return np.nan


def clean_idade(value):
    if pd.isna(value):
        return value
    str_val = str(value)
    if str_val.startswith("1900-01-"):
        try:
            return pd.to_datetime(str_val).day
        except (ValueError, TypeError):
            return np.nan
    try:
        return int(float(str_val))
    except (ValueError, TypeError):
        return value


def clean_genero(value):
    map_generos = {"Menina": "F", "Menino": "M", "Feminino": "F", "Masculino": "M"}
    if pd.isna(value):
        return np.nan
    return map_generos.get(value)


def clean_ra(value):
    if pd.isna(value):
        return np.nan
    str_val = str(value).upper().strip()
    match = re.search(r"(\d+)", str_val)
    if match:
        return int(match.group(1))
    return np.nan


def clean_pedra(value):
    if pd.isna(value) or value == "INCLUIR":
        return None
    if value == "Agata":
        return "Ágata"
    return value


def clean_inde(value):
    if pd.isna(value) or value == "INCLUIR":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def clean_instituicao(value):
    map_instituicao = {
        "Escola Pública": "Pública",
        "Pública": "Pública",
        "Rede Decisão": "Privada",
        "Escola JP II": "Privada",
        "Privada": "Privada",
        "Privada - Programa de Apadrinhamento": "Bolsista",
        "Privada - Programa de apadrinhamento": "Bolsista",
        "Privada *Parcerias com Bolsa 100%": "Bolsista",
        "Privada - Pagamento por *Empresa Parceira": "Bolsista",
        "Bolsista Universitário *Formado (a)": "Bolsista",
        "Concluiu o 3º EM": "Outros",
        "Nenhuma das opções acima": "Outros",
    }
    if pd.isna(value):
        return "Outros"
    return map_instituicao.get(value, "Outros")