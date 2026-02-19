from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any, Tuple

def evaluate_model(y_true, y_pred) -> Tuple[Dict[str, float], str]:
    """
    Calculate metrics for the model using classification_report.
    
    Returns:
        metrics: Dictionary of scalar metrics for MLflow logging.
        report_text: String representation of the classification report for readability.
    """
    # Get dict for MLflow logging
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    # Get string for printing/logging as artifact
    report_text = classification_report(y_true, y_pred)
    
    metrics = {
        "accuracy": report_dict['accuracy'],
        "f1_macro": report_dict['macro avg']['f1-score'],
        "precision_macro": report_dict['macro avg']['precision'],
        "recall_macro": report_dict['macro avg']['recall'],
    }
    
    return metrics, report_text
