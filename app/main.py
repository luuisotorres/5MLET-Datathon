import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import mlflow.pyfunc

# Import our dynamic configurations
from app.config import settings

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    API lifespan manager.
    Attempts to load the model. If it fails, it warns gracefully without crashing the server.
    """
    logger.info("Starting the Passos Mágicos API...")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.model_name}@{settings.model_alias}"

    try:
        logger.info(f"Attempting to load model from URI: {model_uri}")
        app.state.model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model loaded successfully! API is ready for inference.")

    except Exception as e:
        # Graceful degradation: catch the error and keep the server alive
        logger.warning("⚠️ " + "=" * 60)
        logger.warning(
            f"⚠️ WARNING: No model found with the alias '@{settings.model_alias}'."
        )
        logger.warning(
            "⚠️ The API will continue running, but prediction endpoints will fail."
        )
        logger.warning("⚠️ STEPS TO RESOLVE:")
        logger.warning("⚠️ 1. Run the training pipeline (`make train`).")
        logger.warning(
            f"⚠️ 2. Access the MLflow UI and add the '{settings.model_alias}' alias to the best version."
        )
        logger.warning("⚠️ " + "=" * 60)

        # Set model to None so endpoints know it is unavailable
        app.state.model = None

    yield

    logger.info("Shutting down the API and cleaning up memory resources...")
    app.state.model = None


# Application Instance
app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)

# --- Health Check Endpoints ---


@app.get("/")
async def root():
    """
    Root endpoint for health checks.
    Allows reviewers to verify if the web server is up and running.
    """
    model_status = (
        "Loaded and Operational"
        if app.state.model is not None
        else "Awaiting Training/Promotion"
    )

    return {
        "api_name": settings.api_title,
        "version": settings.api_version,
        "status": "Online",
        "model_status": model_status,
        "message": "Welcome to the Passos Mágicos project API.",
    }
