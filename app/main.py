import logging
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI
import mlflow.pyfunc

# Centralized router import
from app.routes import router as api_router

from app.services.model_provider import MLflowModelProvider
from app.services.model_service import ModelService
from app.services.feature_service import FeatureService

# Import our dynamic configurations
from app.config import settings
from passos_magicos.core.paths import ProjectPaths as PP

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Services
provider = MLflowModelProvider()
model_service = ModelService(provider)
feature_service = FeatureService()

# --- Swagger Description (Markdown) ---
description = """
# Passos Mágicos - Student Lagging Risk API 🎓

## Overview
This API was developed to identify the risk of academic lagging for students at the **Passos Mágicos Association**. 
Using Machine Learning models (Random Forest, LightGBM, XGBoost, etc.), the system analyzes psychosocial and learning indicators to predict academic status.

## Features
* **General (Health Check):** System status and connectivity verification.
* **ML Management (Train):** Triggers asynchronous model training. This stage only registers the model in MLflow for evaluation.
* **ML Management (Model):** Discovery endpoint that returns hyperparameters, version, and metadata of the active model.
* **ML Model (Predict):** Performs risk prediction fetching real-time data from the **SQLite Feature Store** based on Student RA.

## MLOps & Governance
The model lifecycle is managed by **MLflow**, ensuring experiment traceability and promotion to the *Production* stage.

---
**Project developed for the Datathon - Post-Graduate Program in Machine Learning and AI**

**Team:**
- Izabelly de Oliveira Menezes
- Larissa Diniz da Silva
- Luis Fernando Torres
- Rafael Dos Santos Callegari
- Renato Massamitsu Zama Inomata
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    API lifespan manager.
    Attempts to load the model from the provider and the student database from disk. 
    """
    logger.info("Starting the Passos Mágicos API...")

    # Inject services into app state
    app.state.model_service = model_service
    app.state.feature_service = feature_service

    # --- Step 1: Load Model ---
    try:
        app.state.model = model_service.load_active_model()
        if app.state.model:
            logger.info("✅ Model loaded successfully! API is ready for inference.")
        else:
            logger.warning("⚠️ Model could not be loaded on startup.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        app.state.model = None

    # --- Step 2: Load Student Database (Fallback Cache) ---
    try:
        data_path = PP.GOLD_DIR / PP.TRAINING_DATA_PARQUET_NAME
        logger.info(f"Attempting to load student database from: {data_path}")
        
        app.state.data = pd.read_parquet(data_path)
        logger.info(f"✅ Student database loaded successfully! ({len(app.state.data)} records)")
        
    except Exception as e:
        logger.warning(f"⚠️ WARNING: Could not load student database: {e}")
        app.state.data = None

    yield

    # Shutdown logic
    logger.info("Shutting down the API and cleaning up memory resources...")
    app.state.model = None
    app.state.data = None


# --- Application Instance ---
app = FastAPI(
    title="Passos Mágicos API", 
    description=description,
    version=settings.api_version, 
    lifespan=lifespan,
    openapi_tags=[
        {"name": "General", "description": "Utility endpoints and system status."},
        {"name": "ML Management", "description": "Model lifecycle operations: Training, Discovery and Reloading."},
        {"name": "ML Model", "description": "Core inference endpoints for lagging risk prediction."}
    ]
)

app.include_router(api_router)

