import logging
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from app.config import settings
from app.requests import (
    HealthCheckResponse,
    ModelMetadataResponse,
    TrainingResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/",
    tags=["General"],
    response_model=HealthCheckResponse,
    summary="API Health Check",
    operation_id="root_health_check",
)
async def root(request: Request):
    """
    Root endpoint for health checks.
    """
    model_status = (
        "Loaded and Operational"
        if getattr(request.app.state, "model", None) is not None
        else "Awaiting Training/Promotion"
    )

    data_status = (
        f"Loaded ({len(request.app.state.data)} records)"
        if getattr(request.app.state, "data", None) is not None
        else "Not Loaded"
    )

    return {
        "api_name": "Passos Mágicos - Student Lagging Risk",
        "version": settings.api_version,
        "status": "Online",
        "model_status": model_status,
        "data_status": data_status,
        "message": "Welcome to the Passos Mágicos Project API.",
    }


@router.post("/train", tags=["ML Management"], response_model=TrainingResponse)
async def train_model(request: Request, background_tasks: BackgroundTasks):
    """
    Triggers the model training pipeline in the background.
    """
    model_service = request.app.state.model_service
    background_tasks.add_task(model_service.train)

    return {
        "status": "Started",
        "message": "Training initiated. The new model will be available in MLflow for evaluation.",
    }


@router.post("/model/reload", tags=["ML Management"])
async def reload_model(request: Request):
    """
    Reloads the model from the provider without restarting the API.
    """
    model_service = request.app.state.model_service
    try:
        new_model = model_service.load_active_model()
        if new_model:
            request.app.state.model = new_model
            version = model_service.get_model_version()
            return {"status": "Success", "message": "Model reloaded successfully.", "version": version}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model from provider.")
    except Exception as e:
        logger.exception("Reload error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model", tags=["ML Management"], response_model=ModelMetadataResponse)
def get_active_model_info(request: Request):
    model = getattr(request.app.state, "model", None)
    model_service = request.app.state.model_service

    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded.")

    metadata = model_service.get_model_metadata(model)
    current_version = model_service.get_model_version()

    return {
        "model_name": settings.model_name,
        "model_alias": settings.model_alias,
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
        "active_metadata": metadata,
        "model_version": current_version,
    }


@router.post("/predict/{ra}", tags=["ML Model"], response_model=PredictionResponse)
def predict_by_ra(ra: str, request: Request):
    """
    Performs prediction based on Student RA. Fetches data from the Feature Store (SQLite).
    """
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    feature_service = request.app.state.feature_service
    features = feature_service.get_student_features(ra)

    if features is None:
        raise HTTPException(
            status_code=404, detail=f"RA {ra} not found in Feature Store."
        )

    try:
        # Inference
        prediction = request.app.state.model.predict(features)
        prediction_value = int(prediction[0])

        # Mapping to populate the 'category' field
        target_map = {
            0: "Critical (Severe Lagging)",
            1: "Alert (Lagging Risk)",
            2: "Expected Performance",
        }

        category_msg = target_map.get(prediction_value, "Unknown Status")

        return {
            "ra": ra,
            "prediction_code": prediction_value,
            "category": category_msg,
            "status": "Success",
            "metadata": {
                "model": settings.model_name,
                "alias": settings.model_alias,
                "source": "SQLite Feature Store",
            },
        }
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")
