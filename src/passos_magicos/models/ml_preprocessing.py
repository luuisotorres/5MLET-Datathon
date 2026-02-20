import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

def create_target_class(y: float) -> int:
    """
    Categorize target_defasagem:
    - value <= -2: critical (0)
    - value == -1: alert (1)
    - value >= 0: expected (2)
    """
    if y <= -2:
        return 0
    if y == -1:
        return 1
    return 2

def clip_indicators(X: pd.DataFrame, min_val: float = 0, max_val: float = 10) -> pd.DataFrame:
    """Clip indicator columns to a specified range."""
    X_clipped = X.copy()
    indicators = [c for c in X.columns if 'indicador_' in c] # Clips only indicator columns
    if indicators:
        X_clipped[indicators] = X_clipped[indicators].clip(min_val, max_val)
    return X_clipped

def map_gender(X: pd.DataFrame) -> pd.DataFrame:
    """Map Gender M/F to 1/0."""
    X_out = X.copy()
    if 'genero' in X_out.columns:
        X_out['genero'] = X_out['genero'].map({'M': 1, 'F': 0})
    return X_out

def get_preprocessor(config: dict) -> ColumnTransformer:
    """
    Build the preprocessing pipeline from config.
    """
    cat_features = config['features']['categorical']
    num_features = config['features']['numerical']
    bin_features = config['features'].get('binary', [])
    
    # -- Transformers --
    
    # 1. Numerical Pipeline
    clipper = FunctionTransformer(
        clip_indicators, 
        kw_args={'min_val': config['preprocessing'].get('clip_min', 0), 
                 'max_val': config['preprocessing'].get('clip_max', 10)},
        validate=False
    )

    # 2. Key Scalers (Optional)
    scaler_type = config['preprocessing'].get('scaler')
    scaler = None
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
        
    # 3. Key Imputers (Optional, we originally dropped NaNs)
    imputer_strategy = config['preprocessing'].get('imputer_strategy')
    
    num_transformer_steps = []
    
    if imputer_strategy:
        num_transformer_steps.append(('imputer', SimpleImputer(strategy=imputer_strategy)))
    
    num_transformer_steps.append(('clipper', clipper))
    
    if scaler:
        num_transformer_steps.append(('scaler', scaler))

    num_pipeline = Pipeline(steps=num_transformer_steps)

    # 4. Categorical Pipeline
    # Using simple OneHotEncoder for categorical features like 'instituicao'
    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # 5. Binary Pipeline
    # Manual mapping for binary features like 'genero'
    bin_transformer = FunctionTransformer(map_gender, validate=False)

    transformers = [
        ('num', num_pipeline, num_features),
        ('cat', cat_transformer, cat_features)
    ]
    
    if bin_features:
        transformers.append(('bin', bin_transformer, bin_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop' 
    )
    
    return preprocessor
