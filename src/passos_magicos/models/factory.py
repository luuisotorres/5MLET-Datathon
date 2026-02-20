from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class ModelFactory:
    "Factory class to instantiate machine learning models"

    @staticmethod
    def get_model(model_type: str, params: dict = None):
        """
        Creates and returns a model instance based on the model_type.
        
        Args:
            model_type (str): The identifier for the model (e.g., 'random_forest').
            params (dict): Hyperparameters for the model.
            
        Returns:
            An instantiated scikit-learn compatible classifier.
            
        Raises:
            ValueError: If the model_type is not supported.
        """
        if params is None:
            params = {}

        if model_type == "random_forest":
            if 'random_state' not in params:
                params['random_state'] = 42
            return RandomForestClassifier(**params)
        elif model_type == "xgboost":
            if 'random_state' not in params:
                params['random_state'] = 42
            return XGBClassifier(**params)
        elif model_type == "lightgbm":
            if 'random_state' not in params:
                params['random_state'] = 42
            return LGBMClassifier(**params)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}"
                f"Supported model types: {['random_forest', 'xgboost', 'lightgbm']}"
            )
            
