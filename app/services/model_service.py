import subprocess
import logging
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from app.config import settings

# Logging configuration
logger = logging.getLogger(__name__)

class ModelService:
    """
    Service responsible for handling the Machine Learning lifecycle:
    Training, Promotion, Feature Preparation, and Metadata Extraction.
    """

    @staticmethod
    def train_and_promote():
        """
        Executes the training pipeline and promotes the new version to 'Production'.
        """
        try:
            logger.info("Starting background training process via 'make train'...")
            
            # Step 1: Execute training via subprocess
            subprocess.run(["make", "train"], check=True, capture_output=True, text=True)
            logger.info("Training script executed successfully.")

            # Step 2: Initialize MLflow Client
            client = MlflowClient()
            model_name = settings.model_name
            
            # Step 3: Get the latest version registered (None stage)
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if not latest_versions:
                raise Exception(f"No new version found for model {model_name}")
                
            new_version = latest_versions[0].version
            logger.info(f"New version detected: {new_version}")

            # Step 4: Archive current Production version
            current_prod = client.get_latest_versions(model_name, stages=["Production"])
            for version in current_prod:
                logger.info(f"Archiving old version: {version.version}")
                client.transition_model_version_stage(
                    name=model_name, version=version.version, stage="Archived"
                )

            # Step 5: Promote new version to Production
            client.transition_model_version_stage(
                name=model_name, version=new_version, stage="Production"
            )
            logger.info(f"✅ Model version {new_version} promoted to Production.")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with exit code {e.returncode}")
            logger.error(f"Subprocess stderr: {e.stderr}")
        except Exception as e:
            logger.exception(f"Unexpected error in automation: {str(e)}")

    @staticmethod
    def prepare_student_features(df_students: pd.DataFrame, ra: str):
        """
        Filters the database to return features for a specific student (RA).
        """
        student_data = df_students[df_students['RA'].astype(str) == str(ra)]
        
        if student_data.empty:
            return None

        required_features = [
            'RA', 'ano_dados', 'fase', 'idade', 'genero', 
            'anos_na_instituicao', 'instituicao', 'inde_atual', 
            'indicador_auto_avaliacao', 'indicador_engajamento', 
            'indicador_psicossocial', 'indicador_aprendizagem', 
            'indicador_ponto_virada', 'indicador_adequacao_nivel', 
            'indicador_psico_pedagogico'
        ]
        
        return student_data[required_features].tail(1)

    @staticmethod
    def get_model_metadata(model):
        """
        Extracts internal hyperparameters and MLflow metadata.
        Standardized to match Tech Challenge 4 discovery endpoints.
        """
        if model is None:
            return None

        try:
            # Step 1: Deep Unwrap
            # Tentamos encontrar o modelo real dentro de diferentes tipos de envelopes do MLflow
            underlying_model = model
            run_id = "N/A"

            # Se for um wrapper do MLflow, tentamos acessar o modelo original
            if hasattr(model, "_model_impl"):
                run_id = getattr(model._model_impl, "run_id", "N/A")
                
                # Caso 1: SklearnModelWrapper (o seu erro atual)
                if hasattr(model._model_impl, "sklearn_model"):
                    underlying_model = model._model_impl.sklearn_model
                # Caso 2: PythonModel wrapper
                elif hasattr(model._model_impl, "python_model"):
                    if hasattr(model._model_impl.python_model, "model"):
                        underlying_model = model._model_impl.python_model.model
                    else:
                        underlying_model = model._model_impl.python_model
            
            # Step 2: Extract details
            if hasattr(underlying_model, 'steps'):
                # É um Pipeline do Scikit-Learn
                step_name, estimator = underlying_model.steps[-1]
                algorithm_name = type(estimator).__name__
                params = estimator.get_params()
                pipeline_steps = [s[0] for s in underlying_model.steps]
            elif hasattr(underlying_model, 'get_params'):
                # É um estimador direto
                algorithm_name = type(underlying_model).__name__
                params = underlying_model.get_params()
                pipeline_steps = []
            else:
                # Fallback caso não consiga extrair params
                algorithm_name = type(underlying_model).__name__
                params = {"info": "Parameters not accessible"}
                pipeline_steps = []

            return {
                "run_id": run_id,
                "algorithm": algorithm_name,
                "pipeline_layers": pipeline_steps,
                "model_params": params,
                "tracking_status": "active_in_production"
            }

        except Exception as e:
            logger.error(f"Failed to extract detailed model metadata: {str(e)}")
            return {
                "algorithm": "Unknown",
                "error": "Extraction failed",
                "details": str(e)
            }