import sqlite3
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

class FeatureService:
    """
    Service for interacting with the Feature Store (SQLite).
    """

    def __init__(self, db_path: str = "feature_store_online.db"):
        self.db_path = db_path

    def get_student_features(self, ra: str) -> Optional[pd.DataFrame]:
        """
        Queries the SQLite Feature Store to return data for a specific student (RA).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            # Querying the 'aluno_features' table
            query = "SELECT * FROM aluno_features WHERE RA = ?"
            student_data = pd.read_sql_query(query, conn, params=(str(ra),))
            conn.close()

            if student_data.empty:
                logger.warning(f"RA {ra} not found in the SQLite Feature Store.")
                return None

            # Feature list must match the model's expected input schema
            required_features = [
                "RA", "ano_dados", "fase", "idade", "genero",
                "anos_na_instituicao", "instituicao", "inde_atual",
                "indicador_auto_avaliacao", "indicador_engajamento",
                "indicador_psicossocial", "indicador_aprendizagem",
                "indicador_ponto_virada", "indicador_adequacao_nivel",
                "indicador_psico_pedagogico",
            ]

            return student_data[required_features].tail(1)

        except Exception as e:
            logger.error(f"Error accessing SQLite Feature Store: {str(e)}")
            return None
