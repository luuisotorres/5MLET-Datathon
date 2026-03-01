import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from app.config import settings

logger = logging.getLogger(__name__)

class ModelProvider(ABC):
    """
    Abstract base class for model providers.
    Defines the interface for loading models and extracting metadata.
    """

    @abstractmethod
    def load_model(self) -> Any:
        """Loads and returns the model object."""
        pass

    @abstractmethod
    def get_metadata(self, model: Any) -> Dict[str, Any]:
        """Extracts metadata from the provided model."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Returns the current version of the model."""
        pass


class MLflowModelProvider(ModelProvider):
    """
    MLflow-specific model provider.
    """

    def __init__(self):
        self.tracking_uri = settings.mlflow_tracking_uri
        self.model_name = settings.model_name
        self.model_alias = settings.model_alias
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def load_model(self) -> Any:
        model_uri = f"models:/{self.model_name}@{self.model_alias}"
        logger.info(f"Attempting to load model from MLflow URI: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)

    def get_version(self) -> str:
        try:
            model_version_details = self.client.get_model_version_by_alias(
                name=self.model_name, 
                alias=self.model_alias
            )
            return model_version_details.version if model_version_details else "N/A"
        except Exception as e:
            logger.warning(f"Could not retrieve model version from MLflow: {e}")
            return "N/A"

    def get_metadata(self, model: Any) -> Dict[str, Any]:
        """
        Extracts hyperparameters and technical metadata from the loaded MLflow model.
        """
        if model is None:
            return {"error": "No model provided"}

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
