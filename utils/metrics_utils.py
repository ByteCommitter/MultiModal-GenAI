import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_fscore_support, confusion_matrix, accuracy_score
from scipy.stats import pearsonr

def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics for PHQ8 prediction"""
    # Clean data
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan'),
            'pearson_r': float('nan'),
            'pearson_p': float('nan')
        }
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p
    }

def compute_classification_metrics(y_true, y_pred):
    """Compute classification metrics for emotion prediction"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def print_metrics_summary(reg_metrics=None, cls_metrics=None):
    """Print a formatted summary of metrics"""
    print("\n=== Evaluation Metrics ===")
    
    if reg_metrics:
        print("\nRegression Metrics (PHQ8):")
        print(f"RMSE: {reg_metrics['rmse']:.4f}")
        print(f"MAE: {reg_metrics['mae']:.4f}")
        print(f"R²: {reg_metrics['r2']:.4f}")
        print(f"Pearson r: {reg_metrics['pearson_r']:.4f} (p={reg_metrics['pearson_p']:.4f})")
    
    if cls_metrics:
        print("\nClassification Metrics (Emotion):")
        print(f"Accuracy: {cls_metrics['accuracy']:.4f}")
        print(f"Precision: {cls_metrics['precision']:.4f}")
        print(f"Recall: {cls_metrics['recall']:.4f}")
        print(f"F1-score: {cls_metrics['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(cls_metrics['confusion_matrix'])
