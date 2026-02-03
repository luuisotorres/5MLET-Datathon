from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class ModelFactory:
    """
    Factory class to create machine learning model instances based on the specified type.
    """
    @staticmethod
    def get_model(model_type, **params):
        if model_type == "random_forest":
            return RandomForestClassifier(**params)
        elif model_type == "logistic_regression":
            return LogisticRegression(**params)
        elif model_type == "decision_tree":
            return DecisionTreeClassifier(**params)
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(**params)
        else:
            raise ValueError(
                f"Model '{model_type}' not supported by the factory.")
