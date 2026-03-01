import subprocess
import logging
from typing import Any, Dict, Optional
from app.services.model_provider import ModelProvider

# Logging Configuration
logger = logging.getLogger(__name__)


class ModelService:
    """
    Service responsible for the model lifecycle:
    Training coordination and Model Loading via Provider.
    """

    def __init__(self, provider: ModelProvider):
        self.provider = provider

    def train(self):
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

    def load_active_model(self) -> Any:
        """
        Loads the active model using the configured provider.
        """
        try:
            return self.provider.load_model()
        except Exception as e:
            logger.error(f"Failed to load model from provider: {e}")
            return None

    def get_model_metadata(self, model: Any) -> Dict[str, Any]:
        """
        Extracts metadata using the provider.
        """
        return self.provider.get_metadata(model)

    def get_model_version(self) -> str:
        """
        Retrieves the model version.
        """
        return self.provider.get_version()
