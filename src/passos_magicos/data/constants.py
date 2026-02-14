from pathlib import Path


class FeatureNames:
    """Centraliza os nomes das colunas da camada Silver/Gold para evitar Magic Strings."""
    RA = 'RA'
    ANO_DADOS = 'ano_dados'
    FASE = 'fase'
    IDADE = 'idade'
    GENERO = 'genero'
    ANO_INGRESSO = 'ano_ingresso'
    ANOS_NA_INSTITUICAO = 'anos_na_instituicao'
    INSTITUICAO = 'instituicao'
    PEDRA_ATUAL = 'pedra_atual'
    INDE = 'inde_atual'
    IAA = 'indicador_auto_avaliacao'
    IEG = 'indicador_engajamento'
    IPS = 'indicador_psicossocial'
    IDA = 'indicador_aprendizagem'
    IPV = 'indicador_ponto_virada'
    IAN = 'indicador_adequacao_nivel'
    IPP = 'indicador_psico_pedagogico'
    DEFASAGEM = 'defasagem'
    TARGET_DEFASAGEM = 'target_defasagem'


class ProjectPaths:
    """Centralizes all directory and file paths for the project."""

    # Base directories
    DATA_DIR = Path("data")

    # Medal layers
    BRONZE_FILE = DATA_DIR / "01_bronze" / "PEDE_2022-24.xlsx"
    SILVER_DIR = DATA_DIR / "02_silver"
    GOLD_DIR = DATA_DIR / "03_gold"

    # Databases
    ONLINE_STORE_DB = Path("feature_store_online.db")
    MLFLOW_DB = Path("mlflow.db")
