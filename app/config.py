import os
from pydantic_settings import BaseSettings
from pydantic import Field


class APISettings(BaseSettings):
    """
    API Configuration Settings.
    Reads from environment variables or uses default values.
    """

    api_title: str = "Passos Mágicos API"
    api_version: str = "1.0.0"

    # MLflow Model Settings
    # This prevents hardcoding "magic strings" in the application logic
    model_name: str = Field(
        default="passos_magicos_defasagem_v1",
        description="The registered name of the model in MLflow",
    )
    model_alias: str = Field(
        default="production",
        description="The MLflow alias to load (e.g., 'production', 'staging')",
    )
    mlflow_tracking_uri: str = Field(
        default="http://127.0.0.1:5000",
        description="URI for the MLflow tracking server",
    )

    class Config:
        # Tells Pydantic to look for a .env file if it exists
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instantiate the settings globally to be imported by the app
settings = APISettings()
