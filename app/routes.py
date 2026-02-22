import logging
import mlflow.pyfunc
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from app.config import settings
from app.services.model_service import ModelService
from app.requests import (
    HealthCheckResponse,
    ModelMetadataResponse, 
    TrainingResponse, 
    PromotionResponse, 
    PredictionResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", 
            tags=["General"], 
            response_model=HealthCheckResponse,
            summary="API Health Check",
            operation_id="root_health_check")
async def root(request: Request):
    """
    Root endpoint for health checks.
    """
    # Acessamos o state através do objeto request
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
async def train_model(background_tasks: BackgroundTasks):
    """
    Triggers the model training pipeline in the background.
    This endpoint only trains and registers the model, leaving promotion as a subsequent manual step.
    """
    # Changed to call training only, without automatic promotion
    background_tasks.add_task(ModelService.train_only) 
    
    return {
        "status": "Started",
        "message": "Training initiated. The new model will be available in MLflow for evaluation."
    }

@router.get("/model", tags=["ML Management"], response_model=ModelMetadataResponse)
def get_active_model_info(request: Request):
    model = getattr(request.app.state, "model", None)
    
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded.")

    metadata = ModelService.get_model_metadata(model)

    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        latest_versions = client.get_latest_versions(settings.model_name, stages=["Production"])
        current_version = latest_versions[0].version if latest_versions else "1"
    except Exception:
        current_version = "N/A" 
    # ---------------------------------------

    return {
        "model_name": settings.model_name,
        "model_alias": settings.model_alias,
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
        "active_metadata": metadata,
        "model_version": current_version
    }

@router.post("/promote/{run_id}", tags=["ML Management"], response_model=PromotionResponse)
async def promote_model_version(run_id: str, request: Request):
    """
    Manual Action: Promotes a specific Run ID to 'Production' in the MLflow Registry
    and updates the model in the API memory (Hot-swap).
    """
    try:
        # 1. Promote in MLflow Registry
        new_version = ModelService.promote_by_run_id(run_id)
        
        # 2. Immediately reload the model into the API memory
        model_uri = f"models:/{settings.model_name}@production"
        request.app.state.model = mlflow.pyfunc.load_model(model_uri)
        
        return {
            "status": "Success",
            "message": f"Model version {new_version} (Run: {run_id}) promoted to Production.",
            "reloaded": True
        }
    except Exception as e:
        logger.error(f"Promotion failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict/{ra}", tags=["ML Model"], response_model=PredictionResponse)
def predict_by_ra(ra: str, request: Request):
    """
    Performs prediction based on Student RA. Fetches data from the Feature Store (SQLite).
    Pydantic validates that the response contains all fields, including 'category'.
    """
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    # Search in SQLite (using 'aluno_features' table as previously adjusted)
    features = ModelService.prepare_student_features_from_db(ra)
    
    if features is None:
        raise HTTPException(status_code=404, detail=f"RA {ra} not found in Feature Store.")

    try:
        # Inference
        prediction = request.app.state.model.predict(features)
        prediction_value = int(prediction[0])

        # Mapping to populate the 'category' field required by Pydantic
        target_map = {
            0: "Critical (Severe Lagging)",
            1: "Alert (Lagging Risk)",
            2: "Expected Performance"
        }
        
        category_msg = target_map.get(prediction_value, "Unknown Status")

        # Return validated by PredictionResponse schema
        return {
            "ra": ra,
            "prediction_code": prediction_value,
            "category": category_msg, 
            "status": "Success",
            "metadata": {
                "model": settings.model_name,
                "alias": settings.model_alias,
                "source": "SQLite Feature Store"
            }
        }
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")