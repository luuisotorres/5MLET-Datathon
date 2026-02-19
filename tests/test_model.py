import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from passos_magicos.config_loader import load_config
from passos_magicos.ml_preprocessing import get_preprocessor
from passos_magicos.dispatcher import MODEL_DISPATCHER

def test_model_dispatcher():
    assert "random_forest" in MODEL_DISPATCHER
    assert "xgboost" in MODEL_DISPATCHER
    assert "lightgbm" in MODEL_DISPATCHER

def test_pipeline_assembly():
    # Mock config
    config = {
        'features': {
            'numerical': ['age', 'score'],
            'categorical': ['group'],
            'binary': ['all_binary_cols_here'] 
        },
        'preprocessing': {
            'clip_min': 0, 
            'clip_max': 10,
            'scaler': None,
            'encoder': 'onehot'
        },
        'model': {
            'type': 'random_forest',
            'params': {'n_estimators': 10}
        }
    }
    
    preprocessor = get_preprocessor(config)
    
    model_class = MODEL_DISPATCHER[config['model']['type']]
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_class(**config['model']['params']))
    ])
    
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == 'preprocessor'
    assert pipeline.steps[1][0] == 'model'
