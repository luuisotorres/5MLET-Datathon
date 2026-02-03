import pandas as pd
from sklearn.datasets import make_classification


def get_dummy_data(n_samples=1000):
    """
    Generic function to generate a dummy classification dataset.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,      
        n_informative=5,
        n_classes=2,
        random_state=42
    )

    cols = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y  

    return df