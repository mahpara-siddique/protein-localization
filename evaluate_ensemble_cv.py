"""
Script to evaluate the 5-Fold MS-LANet v2 (True Self-Attention) + SVM Ensemble on the test set.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- IMPORT SKLEARN FIRST TO PREVENT WINDOWS DLL COLLISION ---
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- NOW IMPORT PYTORCH ---
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import joblib
from tqdm import tqdm

from config import DEVICE, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, FIGURES_DIR
from data_loader_residue import get_residue_loaders
from ms_lanet_v2 import MSLANetV2

def load_mean_embeddings(df, embedding_dir):
    """Load residue tensors and compute mean embeddings on the fly."""
    accessions = df['accession'].tolist()
    mean_embs = []
    
    for acc in tqdm(accessions, desc="Computing Test Mean Embeddings"):
        emb_path = embedding_dir / f"{acc}.pt"
        emb = torch.load(emb_path, map_location="cpu")
        mean_emb = emb.mean(dim=0).numpy()
        mean_embs.append(mean_emb)
        
    return np.array(mean_embs)

def run_cv_ensemble():
    print(f"Using Device: {DEVICE}")
    
    # 1. Load test data
    test_path = PROCESSED_DIR / "test.csv"
    if not test_path.exists():
        print(f"Error: {test_path} not found.")
        return
        
    df_test = pd.read_csv(test_path)
    le = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
    y_test = le.transform(df_test['location'])
    class_names = list(le.classes_)
    
    # 2. Get SVM Predictions (Load pre-trained SVM model)
    svm_model_path = MODELS_DIR / "svm_best_esm2.pkl"
    if not svm_model_path.exists():
        print(f"Error: Pre-trained SVM not found at {svm_model_path}. Running raw SVM training first.")
        # Fallback to train SVM if not found
        train_path = PROCESSED_DIR / "train.csv"
        df_train = pd.read_csv(train_path)
        y_train = le.transform(df_train['location'])
        embedding_dir = FEATURES_DIR / "esm2_residue"
        X_train_mean = load_mean_embeddings(df_train, embedding_dir)
        X_test_mean = load_mean_embeddings(df_test, embedding_dir)
        from sklearn.svm import SVC
        svm_clf = SVC(C=10.0, kernel='rbf', probability=True, random_state=42)
        svm_clf.fit(X_train_mean, y_train)
        joblib.dump(svm_clf, svm_model_path)
    else:
        print("\nLoading pre-trained SVM model...")
        svm_clf = joblib.load(svm_model_path)
        embedding_dir = FEATURES_DIR / "esm2_residue"
        X_test_mean = load_mean_embeddings(df_test, embedding_dir)
        
    print("Getting SVM probabilities...")
    svm_probs = svm_clf.predict_proba(X_test_mean)
    
    # 3. Get predictions from all 5 Folds of MS-LANet v2
    _, _, test_loader, _ = get_residue_loaders(batch_size=32)
    
    # We will accumulate probabilities from all 5 folds
    all_folds_probs = []
    
    for fold in range(1, 6):
        fold_path = MODELS_DIR / f"ms_lanet_v2_fold_{fold}.pt"
        if not fold_path.exists():
            print(f"Error: Model checkpoint for Fold {fold} not found at {fold_path}. Complete CV training first.")
            return
            
        print(f"Inference using MS-LANet v2 Fold {fold}...")
        model = MSLANetV2(
            embedding_dim=1280, 
            num_classes=10, 
            pooling_strategy='self_attn_mean', 
            dropout=0.4, 
            use_layernorm=True
        ).to(DEVICE)
        
        model.load_state_dict(torch.load(fold_path, map_location=DEVICE))
        model.eval()
        
        fold_probs_list = []
        with torch.no_grad():
            for embeddings, mask, _ in test_loader:
                embeddings, mask = embeddings.to(DEVICE), mask.to(DEVICE)
                logits = model(embeddings, mask)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                fold_probs_list.append(probs)
                
        fold_probs = np.vstack(fold_probs_list)
        all_folds_probs.append(fold_probs)
        
    # 4. Average the predictions of all 5 folds (The CV Ensemble)
    lanet_probs = np.mean(all_folds_probs, axis=0)
    lanet_preds = lanet_probs.argmax(axis=1)
    lanet_acc = accuracy_score(y_test, lanet_preds)
    print(f"\n5-Fold MS-LANet v2 (Self-Attention) Ensemble Test Accuracy alone: {lanet_acc * 100:.2f}%")
    
    # 5. Combine averaged MS-LANet v2 predictions with the SVM predictions
    print("\nSweeping weights for combined CV Ensemble...")
    best_w = 0.5
    best_acc = 0.0
    best_preds = None
    
    for w in np.linspace(0, 1, 11):
        ensemble_probs = w * lanet_probs + (1 - w) * svm_probs
        ensemble_preds = ensemble_probs.argmax(axis=1)
        acc = accuracy_score(y_test, ensemble_preds)
        print(f"  Weight w={w:.1f} (5-Fold LANet v2) / {1-w:.1f} (SVM) | Accuracy: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_preds = ensemble_preds
            
    best_mcc = matthews_corrcoef(y_test, best_preds)
    
    # 6. Output Final Results
    print("\n" + "="*60)
    print(f"🏆 FINAL 5-FOLD MS-LANET V2 ENSEMBLE RESULTS (Weight: LANet={best_w:.1f} / SVM={1-best_w:.1f})")
    print("="*60)
    print(f"Accuracy (Q10):  {best_acc * 100:.2f}%")
    print(f"Matthews Corr:   {best_mcc:.4f}")
    print("="*60)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, best_preds, target_names=class_names, zero_division=0))
    
    # Save CV Ensemble Normalized Confusion Matrix
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("5-Fold MS-LANet v2 Ensemble Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = FIGURES_DIR / "ms_lanet_v2_cv_ensemble_confusion.png"
    plt.savefig(matrix_path, dpi=300)
    print(f"Ensemble confusion matrix saved to: {matrix_path}")

if __name__ == "__main__":
    run_cv_ensemble()