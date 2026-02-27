import subprocess
import logging
import sqlite3
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from app.config import settings

# Logging Configuration
logger = logging.getLogger(__name__)


class ModelService:
    """
    Service responsible for the model lifecycle:
    Training and Feature Preparation (via SQLite), and Metadata Extraction.
    """

    @staticmethod
    def train():
        """
        Executes the training pipeline via a subprocess.
        """
        try:
            logger.info("Starting background training process via 'make train'...")

            # Execute training script defined in the Makefile
            result = subprocess.run(
                ["make", "train"], check=True, capture_output=True, text=True
            )

            logger.info("Training script executed successfully.")
            logger.info(f"Subprocess output: {result.stdout}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Training failure: {e.stderr}")
        except Exception as e:
            logger.exception(f"Unexpected error during training: {str(e)}")

    @staticmethod
    def prepare_student_features_from_db(ra: str):
        """
        Queries SQLite Feature Store to return data for a specific student (RA).
        """
        db_path = "feature_store_online.db"
        try:
            conn = sqlite3.connect(db_path)

            # Querying the 'aluno_features' table
            query = "SELECT * FROM aluno_features WHERE RA = ?"
            student_data = pd.read_sql_query(query, conn, params=(str(ra),))
            conn.close()

            if student_data.empty:
                logger.warning(f"RA {ra} not found in the SQLite Feature Store.")
                return None

            # Feature list must match the model's expected input schema
            required_features = [
                "RA",
                "ano_dados",
                "fase",
                "idade",
                "genero",
                "anos_na_instituicao",
                "instituicao",
                "inde_atual",
                "indicador_auto_avaliacao",
                "indicador_engajamento",
                "indicador_psicossocial",
                "indicador_aprendizagem",
                "indicador_ponto_virada",
                "indicador_adequacao_nivel",
                "indicador_psico_pedagogico",
            ]

            return student_data[required_features].tail(1)

        except Exception as e:
            logger.error(f"Error accessing SQLite Feature Store: {str(e)}")
            return None

    @staticmethod
    def get_model_metadata(model):
        """
        Extracts hyperparameters and technical metadata from the loaded MLflow model.
        """
        if model is None:
            return None

        try:
            underlying_model = model
            run_id = "N/A"

            # Navigating through MLflow wrapper to find the core estimator
            if hasattr(model, "_model_impl"):
                run_id = getattr(model._model_impl, "run_id", "N/A")

                if hasattr(model._model_impl, "sklearn_model"):
                    underlying_model = model._model_impl.sklearn_model
                elif hasattr(model._model_impl, "python_model"):
                    if hasattr(model._model_impl.python_model, "model"):
                        underlying_model = model._model_impl.python_model.model
                    else:
                        underlying_model = model._model_impl.python_model

            # Determining algorithm name and parameters
            if hasattr(underlying_model, "steps"):
                step_name, estimator = underlying_model.steps[-1]
                algorithm_name = type(estimator).__name__
                params = estimator.get_params()
                pipeline_steps = [s[0] for s in underlying_model.steps]
            elif hasattr(underlying_model, "get_params"):
                algorithm_name = type(underlying_model).__name__
                params = underlying_model.get_params()
                pipeline_steps = []
            else:
                algorithm_name = type(underlying_model).__name__
                params = {"info": "Parameters not accessible"}
                pipeline_steps = []

            return {
                "run_id": run_id,
                "algorithm": algorithm_name,
                "pipeline_layers": pipeline_steps,
                "model_params": params,
                "tracking_status": "active_in_production",
            }

        except Exception as e:
            logger.error(f"Failed to extract detailed metadata: {str(e)}")
            return {
                "algorithm": "Unknown",
                "error": "Extraction failure",
                "details": str(e),
            }
