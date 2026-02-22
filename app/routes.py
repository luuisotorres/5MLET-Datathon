import logging
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from app.config import settings
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/train", tags=["ML Management"])
async def train_model(background_tasks: BackgroundTasks):
    """
    Triggers the training pipeline and auto-promotes the model to Production.
    Runs in background to avoid timeout.
    """
    background_tasks.add_task(ModelService.train_and_promote)
    return {
        "status": "Started",
        "message": "Training and promotion initiated. Monitor MLflow for updates."
    }

@router.get("/model", tags=["ML Management"])
def get_active_model_info(request: Request):
    """
    Returns the metadata and hyperparameters of the currently active Production model.
    """
    # 1. Access model from app state
    model = getattr(request.app.state, "model", None)
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model currently loaded in the application state."
        )

    # 2. Extract metadata using the service
    metadata = ModelService.get_model_metadata(model)

    return {
        "model_name": settings.model_name,
        "model_alias": settings.model_alias,
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
        "active_metadata": metadata
    }

@router.post("/predict/{ra}", tags=["ML Model"])
def predict_by_ra(ra: str, request: Request):
    """
    Predicts student lagging risk based on their RA.
    """
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="ML Model not loaded.")
    
    df_students = getattr(request.app.state, "data", None)
    if df_students is None:
        raise HTTPException(status_code=500, detail="Student database not loaded.")

    features = ModelService.prepare_student_features(df_students, ra)
    
    if features is None:
        raise HTTPException(status_code=404, detail=f"Student with RA {ra} not found.")

    try:
        prediction = request.app.state.model.predict(features)
        prediction_value = int(prediction[0])

        target_map = {
            0: "Critical (Severe lagging <= -2 years)",
            1: "Alert (Lagging risk = -1 year)",
            2: "Expected (Expected performance >= 0 years)"
        }
        
        category_msg = target_map.get(prediction_value, "Unknown Status")

        return {
            "ra": ra,
            "prediction_code": prediction_value,
            "category": category_msg,
            "status": "Success",
            "metadata": {
                "model": settings.model_name,
                "alias": settings.model_alias
            }
        }
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")