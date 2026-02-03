# src/train.py
import argparse
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from passos_magicos.models import ModelFactory
from passos_magicos.utils import get_dummy_data 

def train(config_path):
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Setup MLflow
    # mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(cfg['experiment_name'])

    with mlflow.start_run(run_name=cfg.get('run_name')):
        # Obtain data
        print("⚠️  Using synthetic data for training.")
        df = get_dummy_data(n_samples=500)
        
        target = cfg['data']['target_column']
        X = df.drop(columns=[target])
        y = df[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg['data']['test_size'], random_state=42
        )

        # Instantiate model
        print(f"🚀 Training model: {cfg['model']['type']}")
        model = ModelFactory.get_model(
            cfg['model']['type'], 
            **cfg['model']['params']
        )

        # Train (fit model)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        print(f"📊 AUC Score: {auc:.4f}")

        # Log metrics and params to MLflow
        mlflow.log_params(cfg['model']['params'])
        mlflow.log_metric("auc", auc)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print("✅ Training completed and artifacts saved in MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)