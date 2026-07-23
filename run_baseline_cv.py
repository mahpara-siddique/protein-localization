"""
5-Fold Cross-Validation for ALL 9 Baseline Models.
Reports mean ± std for Accuracy, Macro-F1, and MCC.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- IMPORT SKLEARN FIRST ---
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from pathlib import Path
import time

# --- IMPORT PYTORCH ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --- IMPORT XGBOOST ---
from xgboost import XGBClassifier

from config import (DEVICE, PROCESSED_DIR, FEATURES_DIR, RESULTS_DIR,
                    RANDOM_SEED, BATCH_SIZE)


# ============================================================
# PYTORCH DL BASELINE MODELS
# ============================================================
class SimpleCNN(nn.Module):
    """1D-CNN baseline for sequence classification."""
    def __init__(self, vocab_size=21, embed_dim=128, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)           # [B, L, 128]
        x = x.transpose(1, 2)           # [B, 128, L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.max(dim=2)[0]             # Global Max Pool → [B, 64]
        x = self.dropout(x)
        return self.fc(x)


class SimpleBiLSTM(nn.Module):
    """BiLSTM baseline for sequence classification."""
    def __init__(self, vocab_size=21, embed_dim=128, hidden_dim=128, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)           # [B, L, 128]
        _, (h_n, _) = self.lstm(x)       # h_n: [4, B, 128]
        # Concatenate last hidden states from both directions
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 256]
        hidden = self.dropout(hidden)
        return self.fc(hidden)


# ============================================================
# HELPER: Train a PyTorch model for one fold
# ============================================================
# ============================================================
# HELPER: Train a PyTorch model for one fold
# ============================================================
def train_pytorch_fold(model, X_train, y_train, X_val, y_val,
                       epochs=20, batch_size=64, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Set up DataLoader for validation to prevent CUDA OOM
    val_ds = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_acc = 0
    best_preds = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate in batches to prevent CUDA OOM
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_logits = model(batch_x)
                preds = batch_logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)

        val_preds = np.array(all_preds)
        val_acc = accuracy_score(y_val, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds = val_preds.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break

    return best_preds


# ============================================================
# HELPER: Compute mean ESM-2 embeddings from .pt files
# ============================================================
def compute_mean_embeddings(df, embedding_dir):
    """Load residue tensors and compute mean embeddings."""
    accessions = df['accession'].tolist()
    mean_embs = []
    for acc in tqdm(accessions, desc="Computing ESM-2 Mean Embeddings"):
        emb = torch.load(embedding_dir / f"{acc}.pt", map_location="cpu")
        mean_embs.append(emb.mean(dim=0).numpy())
    return np.array(mean_embs, dtype=np.float32)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Using Device: {DEVICE}")
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION FOR ALL 9 BASELINE MODELS")
    print("=" * 70)

    # --- 1. Load Training Data & Features ---
    print("\n[1/4] Loading all training features...")
    df_train = pd.read_csv(PROCESSED_DIR / "train.csv")
    labels = np.load(FEATURES_DIR / "train_labels.npy")

    # Classical features (763D) for NB, KNN
    classical = np.load(FEATURES_DIR / "train_combined.npy")
    print(f"  Classical features: {classical.shape}")

    # ESM-2 Mean Embeddings (1280D) for LR, SVM, RF, XGB, MLP
    embedding_dir = FEATURES_DIR / "esm2_residue"
    esm2_mean = compute_mean_embeddings(df_train, embedding_dir)
    print(f"  ESM-2 Mean embeddings: {esm2_mean.shape}")

    # Integer-Encoded Sequences for CNN, BiLSTM
    encoded_seqs = np.load(FEATURES_DIR / "train_encoded.npy")
    print(f"  Encoded sequences: {encoded_seqs.shape}")

    # --- 2. Define All 9 Baselines ---
    print("\n[2/4] Defining baseline models...")

    # (name, model_constructor, feature_array, is_pytorch)
    baselines = [
        # --- ML Baselines on Classical Features ---
        ("Naive Bayes (Classical)",
         lambda: GaussianNB(),
         classical, False),

        ("KNN (Classical)",
         lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
         classical, False),

        # --- ML Baselines on ESM-2 Mean Embeddings ---
        ("Logistic Regression (ESM-2)",
         lambda: LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED, n_jobs=-1),
         esm2_mean, False),

        ("SVM RBF (ESM-2)",
         lambda: SVC(C=10.0, kernel='rbf', random_state=RANDOM_SEED),
         esm2_mean, False),

        ("Random Forest (ESM-2)",
         lambda: RandomForestClassifier(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1),
         esm2_mean, False),

        ("XGBoost (ESM-2)",
         lambda: XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                               random_state=RANDOM_SEED, use_label_encoder=False,
                               eval_metric='mlogloss', n_jobs=-1),
         esm2_mean, False),

        ("MLP (ESM-2)",
         lambda: MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=200,
                               random_state=RANDOM_SEED, early_stopping=True),
         esm2_mean, False),

        # --- DL Baselines on Encoded Sequences ---
        ("1D-CNN (Sequences)",
         lambda: SimpleCNN(vocab_size=21, embed_dim=128, num_classes=10),
         encoded_seqs, True),

        ("BiLSTM (Sequences)",
         lambda: SimpleBiLSTM(vocab_size=21, embed_dim=128, hidden_dim=128, num_classes=10),
         encoded_seqs, True),
    ]

    # --- 3. Run 5-Fold CV ---
    print("\n[3/4] Running 5-Fold Stratified Cross-Validation...\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    all_results = []

    for name, model_fn, features, is_pytorch in baselines:
        print(f"  {'='*60}")
        print(f"  Model: {name}")
        print(f"  {'='*60}")

        fold_accs, fold_f1s, fold_mccs = [], [], []
        start = time.time()

        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
            X_tr, X_val = features[train_idx], features[val_idx]
            y_tr, y_val = labels[train_idx], labels[val_idx]

            if is_pytorch:
                # PyTorch model
                model = model_fn()
                preds = train_pytorch_fold(model, X_tr, y_tr, X_val, y_val,
                                           epochs=20, batch_size=64)
            else:
                # Sklearn model
                model = model_fn()

                # Scale features for models that benefit from normalization
                if name in ["Logistic Regression (ESM-2)", "SVM RBF (ESM-2)",
                            "KNN (Classical)", "MLP (ESM-2)"]:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_val = scaler.transform(X_val)

                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)

            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            mcc = matthews_corrcoef(y_val, preds)
            fold_accs.append(acc)
            fold_f1s.append(f1)
            fold_mccs.append(mcc)
            print(f"    Fold {fold+1}: Acc={acc:.4f} | F1={f1:.4f} | MCC={mcc:.4f}")

        elapsed = time.time() - start

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s)
        mean_mcc = np.mean(fold_mccs)
        std_mcc = np.std(fold_mccs)

        print(f"  ► RESULT: Acc={mean_acc:.4f}±{std_acc:.4f} | "
              f"F1={mean_f1:.4f}±{std_f1:.4f} | "
              f"MCC={mean_mcc:.4f}±{std_mcc:.4f} | ({elapsed:.1f}s)")

        all_results.append({
            'Model': name,
            'Accuracy_Mean': f"{mean_acc:.4f}",
            'Accuracy_Std': f"{std_acc:.4f}",
            'F1_Mean': f"{mean_f1:.4f}",
            'F1_Std': f"{std_f1:.4f}",
            'MCC_Mean': f"{mean_mcc:.4f}",
            'MCC_Std': f"{std_mcc:.4f}",
        })

    # --- 4. Save Results ---
    print("\n[4/4] Saving results...\n")
    results_df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "baseline_cv_results.csv"
    results_df.to_csv(results_path, index=False)

    print("=" * 70)
    print("📊 FINAL BASELINE RESULTS (5-Fold CV, Mean ± Std)")
    print("=" * 70)
    for _, row in results_df.iterrows():
        print(f"  {row['Model']:35s} | "
              f"Acc: {row['Accuracy_Mean']}±{row['Accuracy_Std']} | "
              f"F1: {row['F1_Mean']}±{row['F1_Std']} | "
              f"MCC: {row['MCC_Mean']}±{row['MCC_Std']}")
    print("=" * 70)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()