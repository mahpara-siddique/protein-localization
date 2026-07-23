"""
Training the optimal MS-LANet v2 model using 5-Fold Cross-Validation
and Label Smoothing regularization (discovered via Ablation Studies).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- IMPORT SKLEARN AND NUMPY FIRST ---
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd

# --- NOW IMPORT PYTORCH ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
import joblib

from config import DEVICE, MODELS_DIR, PROCESSED_DIR, FEATURES_DIR, RANDOM_SEED
from data_loader_residue import ResidueEmbeddingDataset
from ms_lanet_v2 import MSLANetV2

def get_fold_loaders(df_train, train_idx, val_idx, batch_size=32):
    """Generate PyTorch DataLoaders for a specific fold."""
    embedding_dir = FEATURES_DIR / "esm2_residue"
    
    df_tr = df_train.iloc[train_idx]
    df_val = df_train.iloc[val_idx]
    
    train_ds = ResidueEmbeddingDataset(df_tr['accession'].tolist(), df_tr['label'].tolist(), embedding_dir)
    val_ds = ResidueEmbeddingDataset(df_val['accession'].tolist(), df_val['label'].tolist(), embedding_dir)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader

def train_cv():
    print(f"Using Device: {DEVICE}")
    
    # 1. Load Data
    train_path = PROCESSED_DIR / "train.csv"
    df_train = pd.read_csv(train_path)
    le = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
    df_train['label'] = le.transform(df_train['location'])
    
    labels = df_train['label'].values
    indices = np.arange(len(df_train))
    
    # 2. Setup 5-Fold Stratified Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    print(f"\nInitialized 5-Fold CV on {len(df_train)} training sequences.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Optimal Hyperparameters from Ablation Study
    lr = 5e-4
    weight_decay = 1e-3
    patience = 6
    max_epochs = 25
    
    # Label Smoothing Cross-Entropy Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n" + "="*70)
        print(f"🚀 TRAINING FOLD {fold + 1} / 5 (Best Config: True Self-Attention + Label Smoothing)")
        print("="*70)
        
        train_loader, val_loader = get_fold_loaders(df_train, train_idx, val_idx, batch_size=32)
        
        # Initialize MS-LANet v2 with true self-attention pooling
        model = MSLANetV2(
            embedding_dim=1280, 
            pooling_strategy='self_attn_mean', 
            dropout=0.4, 
            use_layernorm=True
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        
        fold_model_path = MODELS_DIR / f"ms_lanet_v2_fold_{fold + 1}.pt"
        
        best_val_f1 = 0.0
        epochs_no_improve = 0
        
        for epoch in range(max_epochs):
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for embeddings, mask, batch_labels in train_loader:
                embeddings, mask, batch_labels = embeddings.to(DEVICE), mask.to(DEVICE), batch_labels.to(DEVICE)
                
                logits = model(embeddings, mask)
                loss = criterion(logits, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
            avg_train_loss = train_loss / train_batches
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for embeddings, mask, batch_labels in val_loader:
                    embeddings, mask, batch_labels = embeddings.to(DEVICE), mask.to(DEVICE), batch_labels.to(DEVICE)
                    logits = model(embeddings, mask)
                    loss = criterion(logits, batch_labels)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_labels.cpu().numpy())
                    
            avg_val_loss = val_loss / val_batches
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            print(f"Fold {fold+1} | Ep {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            scheduler.step(val_f1)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                torch.save(model.state_dict(), fold_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Fold {fold+1} early stopped. Best Val F1: {best_val_f1:.4f}")
                    break
                    
        print(f"✓ Fold {fold+1} Complete. Saved model to: {fold_model_path}")
        
    print("\n" + "="*70)
    print("✅ OPTIMIZED 5-FOLD MS-LANET V2 TRAINED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    train_cv()