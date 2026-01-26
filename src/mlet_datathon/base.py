from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin


class BaseModelWrapper(BaseEstimator, ABC):
    """
    Base class for models. 
    """
    
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    # # Create new methods that are not from base sklearn but may be needed in this project
    # def log_metrics(self, y_true, y_pred, step_name="train"):
    #     """
    #     Log metrics in MLFlow.
    #     """
    #     pass

    
class BaseTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Base class for preprocessing and feature engineering.
    """
    
    def fit(self, X, y=None):
        # Most simple transformers don't need a complex fit method.
        return self

    @abstractmethod
    def transform(self, X):
        """
        Transformation logic to be implemented.
        """
        pass

    def get_feature_names_out(self, input_features=None):
        """
        Optional: helps tracking names of columns in the pipeline.
        """
        pass
