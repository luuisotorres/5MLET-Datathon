import pandas as pd
import numpy as np
import re


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
