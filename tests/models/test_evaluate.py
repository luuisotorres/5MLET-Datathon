import pytest
import pandas as pd
from passos_magicos.models.evaluate import evaluate_model

def test_evaluate_model():
    """Test that evaluate_model returns the correct structure (dict, str)."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 0] # Last one is wrong
    
    metrics, report = evaluate_model(y_true, y_pred)
    
    # Check metrics dict keys
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    
    # Check report string
    assert isinstance(report, str)
    assert "precision" in report
    assert "recall" in report
    assert "f1-score" in report
    assert "accuracy" in report
