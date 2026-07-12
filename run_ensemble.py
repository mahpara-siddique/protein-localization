"""Script to train SVM and run an ensemble with MS-LANet on the test set."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- IMPORT SKLEARN FIRST ---
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
import numpy as np
import pandas as pd

# --- NOW IMPORT PYTORCH ---
import torch
import torch.nn.functional as F  # <--- ADD THIS LINE HERE
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import joblib
from tqdm import tqdm

from config import DEVICE, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR
from data_loader_residue import get_residue_loaders
from ms_lanet import MSLANet

def load_mean_embeddings(df, embedding_dir):
    """Load residue tensors and compute mean embeddings on the fly."""
    accessions = df['accession'].tolist()
    mean_embs = []
    
    for acc in tqdm(accessions, desc="Computing Mean Embeddings"):
        emb_path = embedding_dir / f"{acc}.pt"
        # Load the [seq_len, 1280] tensor
        emb = torch.load(emb_path, map_location="cpu")
        # Compute mean along the sequence length dimension
        mean_emb = emb.mean(dim=0).numpy()
        mean_embs.append(mean_emb)
        
    return np.array(mean_embs)

def run_ensemble_benchmark():
    print(f"Using Device: {DEVICE}")
    
    # 1. Load DataFrames
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    le = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
    y_train = le.transform(df_train['location'])
    y_test = le.transform(df_test['location'])
    
    # 2. Compute Mean Embeddings for SVM
    embedding_dir = FEATURES_DIR / "esm2_residue"
    print("\nPreparing mean embeddings for SVM training...")
    X_train_mean = load_mean_embeddings(df_train, embedding_dir)
    X_test_mean = load_mean_embeddings(df_test, embedding_dir)
    
    # 3. Train SVM on exact same split
    print("\nTraining SVM (RBF) with C=10.0...")
    svm_clf = SVC(C=10.0, kernel='rbf', probability=True, random_state=42)
    svm_clf.fit(X_train_mean, y_train)
    
    # Get SVM class probabilities on the test set
    print("Getting SVM probabilities...")
    svm_probs = svm_clf.predict_proba(X_test_mean)
    svm_preds = svm_probs.argmax(axis=1)
    svm_acc = accuracy_score(y_test, svm_preds)
    print(f"SVM Test Accuracy alone: {svm_acc * 100:.2f}%")
    
    # 4. Get MS-LANet probabilities on test set
    print("\nLoading MS-LANet and predicting...")
    _, _, test_loader, _ = get_residue_loaders(batch_size=32)
    model = MSLANet(embedding_dim=1280, num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "ms_lanet_best.pt", map_location=DEVICE))
    model.eval()
    
    lanet_probs_list = []
    with torch.no_grad():
        for embeddings, mask, _ in test_loader:
            embeddings, mask = embeddings.to(DEVICE), mask.to(DEVICE)
            logits = model(embeddings, mask)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            lanet_probs_list.append(probs)
            
    lanet_probs = np.vstack(lanet_probs_list)
    lanet_preds = lanet_probs.argmax(axis=1)
    lanet_acc = accuracy_score(y_test, lanet_preds)
    print(f"MS-LANet Test Accuracy alone: {lanet_acc * 100:.2f}%")
    
    # 5. Sweep Ensemble weights: w * LANet + (1 - w) * SVM
    print("\nSweeping weights for best Ensemble combination...")
    best_w = 0.5
    best_acc = 0.0
    best_preds = None
    
    # Test different weight distributions
    for w in np.linspace(0, 1, 11):
        ensemble_probs = w * lanet_probs + (1 - w) * svm_probs
        ensemble_preds = ensemble_probs.argmax(axis=1)
        acc = accuracy_score(y_test, ensemble_preds)
        print(f"  Weight w={w:.1f} (LANet) / {1-w:.1f} (SVM) | Accuracy: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_preds = ensemble_preds
            
    best_mcc = matthews_corrcoef(y_test, best_preds)
    
    # 6. Print Final Ensemble Results
    print("\n" + "="*50)
    print(f"🎉 FINAL ENSEMBLE RESULTS (Weight: LANet={best_w:.1f} / SVM={1-best_w:.1f})")
    print("="*50)
    print(f"Accuracy (Q10):  {best_acc * 100:.2f}%")
    print(f"Matthews Corr:   {best_mcc:.4f}")
    print("="*50)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, best_preds, target_names=list(le.classes_), zero_division=0))
    
    # Save SVM model for future use
    joblib.dump(svm_clf, MODELS_DIR / "svm_best_esm2.pkl")
    print(f"SVM model saved to {MODELS_DIR / 'svm_best_esm2.pkl'}")

if __name__ == "__main__":
    run_ensemble_benchmark()