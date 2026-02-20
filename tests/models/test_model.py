import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from passos_magicos.models.config_loader import load_config
from passos_magicos.models.ml_preprocessing import get_preprocessor
from passos_magicos.models.factory import ModelFactory

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

    model_name = config['model']['type']
    model_params = config['model']['params']
    
    classifier = ModelFactory.get_model(model_name, model_params)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', classifier)
    ])
    
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == 'preprocessor'
    assert pipeline.steps[1][0] == 'model'
