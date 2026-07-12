"""
Evaluation metrics and visualization tools.
Calculates Q10 Accuracy, Macro-F1, and Matthews Correlation Coefficient (MCC).
Generates confusion matrix plots.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
from config import FIGURES_DIR


def evaluate_predictions(y_true, y_pred, label_names=None, model_name="Model"):
    """Calculate and print classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print("\n" + "=" * 50)
    print(f"RESULTS FOR: {model_name}")
    print("=" * 50)
    print(f"Accuracy (Q10): {acc:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    print(f"MCC:            {mcc:.4f}")
    print("=" * 50)
    
    if label_names is not None:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=label_names))
        
    return {'accuracy': acc, 'macro_f1': f1, 'mcc': mcc}


def plot_confusion_matrix(y_true, y_pred, label_names, filename):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalized
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues", 
        xticklabels=label_names, yticklabels=label_names, 
        ax=ax, cbar=True, square=True
    )
    ax.set_title(f"Confusion Matrix", fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved confusion matrix heatmap → {save_path}")