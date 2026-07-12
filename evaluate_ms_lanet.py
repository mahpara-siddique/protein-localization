"""Evaluation script to benchmark MS-LANet on the test set."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- IMPORT SKLEARN FIRST ---
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- NOW IMPORT PYTORCH ---
import torch
from pathlib import Path
import sys
import joblib

from config import DEVICE, MODELS_DIR, FIGURES_DIR
from data_loader_residue import get_residue_loaders
from ms_lanet import MSLANet

def evaluate_model():
    print(f"Using Device: {DEVICE}")
    
    # 1. Load Data loaders
    print("Loading test data loader...")
    _, _, test_loader, _ = get_residue_loaders(batch_size=32)
    
    # 2. Load Model
    model = MSLANet(embedding_dim=1280, num_classes=10).to(DEVICE)
    model_path = MODELS_DIR / "ms_lanet_best.pt"
    
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Loaded best MS-LANet model weights.")
    
    # 3. Load Label Encoder to get text names for classes
    le = joblib.load(Path("data/processed/label_encoder.pkl"))
    class_names = list(le.classes_)
    
    all_preds = []
    all_labels = []
    
    print("\nRunning inference on test set (4308 sequences)...")
    with torch.no_grad():
        for embeddings, mask, labels in test_loader:
            embeddings, mask = embeddings.to(DEVICE), mask.to(DEVICE)
            
            logits = model(embeddings, mask)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    # 4. Compute Metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print("\n" + "="*50)
    print("📊 MS-LANet TEST SET EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy (Q10):  {accuracy * 100:.2f}%")
    print(f"Matthews Corr:   {mcc:.4f}")
    print("="*50)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # 5. Plot Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix by rows (true classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("MS-LANet Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = FIGURES_DIR / "ms_lanet_confusion.png"
    plt.savefig(matrix_path, dpi=300)
    print(f"Confusion matrix saved to: {matrix_path}")

if __name__ == "__main__":
    evaluate_model()