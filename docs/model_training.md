# Model Training Documentation

The training environment for the Passos Mágicos student lagging prediction model is designed for experiment tracking, performance monitoring, and model promotion for production deployment.

## 1. MLOps Ecosystem (MLflow)

All training runs are automatically tracked using **MLflow**. The lifecycle includes:
- **Tracking**: Logs all hyperparameters (from `config.yaml`), training metrics, and artifacts.
- **Model Registry**: Stores the best models as versioned entities (`passos_magicos_defasagem_v1`).
- **Artifacts**: Stores the exact configuration file and classification report for every run.

**To access the UI during training**:
```bash
make ui
```
*Accessible at `http://localhost:5000`*.

---

## 2. Training Strategy

### 🔎 Data Splitting (Time-Based)
The dataset is split by year (not randomly row by row). This approach prevents **data leakage** and reflects how the model will be used in the real world:
- **Training Set**: All student data from 2022.
- **Test Set**: All student data from 2023.

*This allows us to evaluate the model's ability to predict the next year's outcomes.*

### 🎯 Target Categorization (Defasagem)

The raw `target_defasagem` represents the academic delay in years (e.g., -2 means a student is 2 years behind their expected grade). For the machine learning model, this is discretized into three priority classes using the `create_target_class` logic:

| Class | Category | Rule (Shifted Defasagem) | Description |
| :--- | :--- | :--- | :--- |
| **0** | 🔴 **Critical** | `value <= -2` | Severe lagging. Priority 1 for pedagogical intervention. |
| **1** | 🟡 **Alert** | `value == -1` | Early sign of falling behind. Monitoring required. |
| **2** | 🟢 **Expected** | `value >= 0` | Student is at or ahead of their expected academic level. |

---

### 🧼 Preprocessing
The model uses a Scikit-Learn **Pipeline** to ensure that all transformations applied during training are identical during inference:
- **Numerical Features**: Scaling (StandardScaler) + Simple Imputer (Mean).
- **Categorical Features**: One-Hot Encoding (OHE) + Simple Imputer (Constant).
- **Clipping**: Indicator values (INDE, IPP, etc.) are clipped within the [0, 10] range to avoid outlier influence.

---

## 3. Supported Model Architectures

The training script (`train.py`) uses a **Model Factory** to support different ensemble algorithms:

1.  **Random Forest** (Default): Robust against noise and good for initial baselines.
2.  **XGBoost**: High performance with gradient boosting, optimized for speed.
3.  **LightGBM**: Efficient gradient boosting for larger datasets.

**Configuration (`config/config.yaml`)**:
```yaml
model:
  type: "random_forest"
  params:
    n_estimators: 200
    max_depth: 8
    class_weight: "balanced"  # Crucial due to imbalanced lagging classes
```

---

## 4. Evaluation Metrics

Every model run generates a comprehensive classification report logged to MLflow:
- **Accuracy**: Overall correctness.
- **Recall (per class)**: Extremely important for the "Critical" class—we cannot afford to miss students at high risk.
- **Precision**: Ensures the NGO doesn't waste resources on false positives.
- **F1-Score**: Harmonic mean of Precision and Recall.

---

## 5. Model Promotion (Production)

To allow the FastAPI application to consume a new model, it **must** be promoted in the MLflow Registry.

1.  **Find the Best Version**: Identify the run with the highest Recall for high-risk students.
2.  **Assign Alias**: Go to the **Model Registry** tab, select the version, and add the alias `production`.
3.  **Hot Reload**: The API can be reloaded without a restart:
    ```bash
    curl -X POST http://localhost:8000/model/reload
    ```

---

## 6. Recommendations for Future Model Training

To further evolve the predictive capabilities of the platform, the following training improvements are suggested:

1. **Automated Hyperparameter Tuning**: Transition from manual parameter setting to automated search using **Optuna** or **Ray Tune** to find optimal model configurations across different architectures.
2. **Cross-Validation Expansion**: Move from a single time-series split to **Time-Series Cross-Validation**. This provides more robust performance estimation by training/testing across multiple rolling windows of years.
3. **Advanced Feature Engineering**: Explore derived indices such as "Academic Momentum" (rate of change in INDE over years) or psychosocial "Alert Flags" based on specific historical drops in attendance or engagement.
4. **Model Interpretability**: Integrate **SHAP (SHapley Additive exPlanations)** or **LIME** inside the MLflow artifacts to explain WHY the model categorized a student as high-risk, assisting the NGO's educators in focused interventions.
5. **Continuous Performance Monitoring**: Implement real-time monitoring of model precision/recall in production to detect model decay and trigger periodic retrains when accuracy falls below a set threshold.
