import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from pprint import pprint

from passos_magicos.config_loader import load_config
from passos_magicos.dispatcher import MODEL_DISPATCHER
from passos_magicos.ml_preprocessing import create_target_class, get_preprocessor
from passos_magicos.evaluate import evaluate_model

def main():
    # 1. Load Configuration
    config = load_config("config.yaml")
    print("Configuration loaded:")
    pprint(config)

    # 2. Setup MLflow
    mlflow.set_experiment(config['experiment_name'])
    
    with mlflow.start_run(run_name=config.get('run_name', 'default_run')) as run:
        # Log params (flattened for convenience)
        mlflow.log_params(config['model']['params'])
        mlflow.log_param("model_type", config['model']['type'])
        
        # 3. Load Data
        df = pd.read_parquet(config['data']['input_path'])
        
        # 4. Preprocessing (Data Cleaning & Feature Engineering)
        df_clean = df.dropna().copy()
        
        # Create Target Class
        if 'target_defasagem' in df_clean.columns:
             df_clean['target_class'] = df_clean['target_defasagem'].apply(create_target_class)
        
        # Drop irrelevant columns for X
        X = df_clean.drop(columns=['defasagem', 'target_defasagem', 'pedra_atual', 'target_class'])
        y = df_clean['target_class']

        # 5. Split Data (Time-based split)
        train_year = config['data'].get('filter_year_train', 2022)
        test_year = config['data'].get('filter_year_test', 2023)
        
        mask_train = X['ano_dados'] == train_year
        mask_test = X['ano_dados'] == test_year
        
        X_train, y_train = X[mask_train].copy(), y[mask_train].copy()
        X_test, y_test = X[mask_test].copy(), y[mask_test].copy()
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # 6. Build Pipeline
        preprocessor = get_preprocessor(config)
        
        model_name = config['model']['type']
        model_class = MODEL_DISPATCHER[model_name]
        model_params = config['model']['params']
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model_class(**model_params))
        ])
        
        # 7. Train
        print("Training model...")
        pipeline.fit(X_train, y_train)
        
        # 8. Evaluate
        print("Evaluating model...")
        y_pred = pipeline.predict(X_test)
        metrics, report = evaluate_model(y_test, y_pred)
        
        print("Classification Report:\n", report)
        print("Metrics:", metrics)
        mlflow.log_metrics(metrics)
        mlflow.log_text(report, "classification_report.txt")
        
        # 9. Log Model
        signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name=config.get('registered_model_name'),
            input_example=X_train.iloc[:5]
        )
        
        # Log config file as artifact
        mlflow.log_artifact("config.yaml")
        
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
