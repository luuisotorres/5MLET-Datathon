from mlet_datathon.base import BaseTransformer


class DropNullsTransformer(BaseTransformer):
    def transform(self, X):
        # Implementação específica
        return X.dropna().copy()


class FillMissingValues(BaseTransformer):
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.fill_values = None

    def fit(self, X, y=None):
        if self.strategy == 'mean':
            self.fill_values = X.mean()
        # ... fit logic
        return self

    def transform(self, X):
        return X.fillna(self.fill_values)

