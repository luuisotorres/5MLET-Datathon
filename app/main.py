import logging
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI
import mlflow.pyfunc

# Centralized router import
from app.routes import router as api_router

# Import our dynamic configurations
from app.config import settings

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Swagger Description (Markdown) ---
description = """
# Passos Mágicos - Student Lagging Risk API 🎓

## Overview
This API was developed to identify the risk of academic lagging for students at the **Passos Mágicos Association**. 
Using Machine Learning models (Random Forest), the system analyzes psychosocial and learning indicators to predict academic status.

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
    Attempts to load the model from MLflow and the student database from disk. 
    """
    logger.info("Starting the Passos Mágicos API...")

    # --- Step 1: Load Model from MLflow ---
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.model_name}@{settings.model_alias}"

    try:
        logger.info(f"Attempting to load model from URI: {model_uri}")
        app.state.model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model loaded successfully! API is ready for inference.")

    except Exception as e:
        logger.warning("⚠️ " + "=" * 60)
        logger.warning(
            f"⚠️ WARNING: No model found with the alias '@{settings.model_alias}'."
        )
        logger.warning(
            "⚠️ The API will continue running, but prediction endpoints will fail."
        )
        logger.warning("⚠️ STEPS TO RESOLVE: Run 'make train' and promote the model in MLflow UI.")
        logger.warning("⚠️ " + "=" * 60)
        app.state.model = None

    # --- Step 2: Load Student Database ---
    try:
        data_path = "data/03_gold/train_data.parquet"
        logger.info(f"Attempting to load student database from: {data_path}")
        
        app.state.data = pd.read_parquet(data_path)
        logger.info(f"✅ Student database loaded successfully! ({len(app.state.data)} records)")
        
    except Exception as e:
        logger.warning("⚠️ " + "=" * 60)
        logger.warning(f"⚠️ WARNING: Could not load student database: {e}")
        logger.warning("⚠️ Ensure you have run 'make_gold.py' to generate the data files.")
        logger.warning("⚠️ " + "=" * 60)
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
        {"name": "ML Management", "description": "Model lifecycle operations: Training and Discovery."},
        {"name": "ML Model", "description": "Core inference endpoints for lagging risk prediction."}
    ]
)

app.include_router(api_router)

