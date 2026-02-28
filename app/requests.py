from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class HealthCheckResponse(BaseModel):
    api_name: str
    version: str
    status: str
    model_status: str
    data_status: str
    message: str


class ModelMetadataResponse(BaseModel):
    model_name: str
    model_alias: str
    mlflow_tracking_uri: str
    active_metadata: dict
    model_version: str


class TrainingResponse(BaseModel):
    status: str
    message: str


class PredictionResponse(BaseModel):
    ra: str
    prediction_code: int
    category: str
    status: str
    metadata: Dict[str, str]
