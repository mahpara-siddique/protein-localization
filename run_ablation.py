"""
Automated Ablation Study Runner for MS-LANet v2.
Runs 12 experiments to systematically justify each architectural design choice.
Saves results to results/ablation_results.csv.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- IMPORT SKLEARN FIRST ---
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import numpy as np
import pandas as pd
import joblib

# --- NOW IMPORT PYTORCH ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path

from config import DEVICE, MODELS_DIR, RESULTS_DIR, BATCH_SIZE
from data_loader_residue import get_residue_loaders
from ms_lanet_v2 import MSLANetV2

# --- Custom Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

def compute_class_weights(labels, num_classes=10):
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * class_counts.clip(min=1))
    return torch.FloatTensor(weights).to(DEVICE)

def train_ablation_run(exp_name, pooling_strategy, loss_type, dropout, use_layernorm, 
                        train_loader, val_loader, test_loader, class_weights):
    print(f"\n==================================================")
    print(f"🚀 Running: {exp_name}")
    print(f"==================================================")
    
    # 1. Initialize Model
    model = MSLANetV2(
        embedding_dim=1280,
        pooling_strategy=pooling_strategy,
        dropout=dropout,
        use_layernorm=use_layernorm
    ).to(DEVICE)
    
    # 2. Select Loss function
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    elif loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'label_smoothing':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )
    
    best_val_f1 = 0.0
    epochs_no_improve = 0
    patience = 5
    best_model_path = MODELS_DIR / f"temp_{exp_name}.pt"
    
    # Train Loop
    for epoch in range(25):
        model.train()
        for embeddings, mask, labels in train_loader:
            embeddings, mask, labels = embeddings.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            logits = model(embeddings, mask)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for embeddings, mask, labels in val_loader:
                embeddings, mask, labels = embeddings.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
                logits = model(embeddings, mask)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    # 3. Test Set Evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for embeddings, mask, labels in test_loader:
            embeddings, mask = embeddings.to(DEVICE), mask.to(DEVICE)
            logits = model(embeddings, mask)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    test_mcc = matthews_corrcoef(all_labels, all_preds)
    
    print(f"✅ Finished: {exp_name}")
    print(f"   Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1:.4f} | Test MCC: {test_mcc:.4f}")
    
    # Cleanup temp checkpoint
    if best_model_path.exists():
        os.remove(best_model_path)
        
    return test_acc, test_f1, test_mcc

def main():
    print(f"Using Device: {DEVICE}")
    
    # Load loaders & compute weights
    train_loader, val_loader, test_loader, y_train = get_residue_loaders(batch_size=32)
    class_weights = compute_class_weights(y_train, num_classes=10)
    
    # Define the 12 Ablation Experiments
    experiments = [
        # (exp_name, pooling_strategy, loss_type, dropout, use_layernorm)
        
        # --- Pooling Mechanisms (1-7) ---
        ("01_mean_only", "mean_only", "focal", 0.4, True),
        ("02_max_only", "max_only", "focal", 0.4, True),
        ("03_mean_max", "mean_max", "focal", 0.4, True),
        ("04_conv_pool_only", "conv_pool_only", "focal", 0.4, True),
        ("05_conv_pool_mean", "conv_pool_mean", "focal", 0.4, True),
        ("06_self_attn_only", "self_attn_only", "focal", 0.4, True),
        ("07_self_attn_mean", "self_attn_mean", "focal", 0.4, True), # Proposed v2
        
        # --- Loss Function (8-9) ---
        ("08_ce_loss", "self_attn_mean", "ce", 0.4, True),
        ("09_label_smoothing", "self_attn_mean", "label_smoothing", 0.4, True),
        
        # --- Regularization (10-12) ---
        ("10_dropout_0.1", "self_attn_mean", "focal", 0.1, True),
        ("11_dropout_0.3", "self_attn_mean", "focal", 0.3, True),
        ("12_no_layernorm", "self_attn_mean", "focal", 0.4, False),
    ]
    
    results = []
    
    for exp in experiments:
        acc, f1, mcc = train_ablation_run(
            *exp, train_loader, val_loader, test_loader, class_weights
        )
        results.append({
            "Experiment": exp[0],
            "Pooling": exp[1],
            "Loss": exp[2],
            "Dropout": exp[3],
            "LayerNorm": exp[4],
            "Test_Acc": f"{acc*100:.2f}%",
            "Test_MacroF1": f"{f1:.4f}",
            "Test_MCC": f"{mcc:.4f}"
        })
        
    # Save Results CSV
    results_df = pd.DataFrame(results)
    results_path = RESULTS_DIR / "ablation_results.csv"
    results_df.to_csv(results_path, index=False)
    
    print("\n" + "="*70)
    print("📊 ABLATION STUDY RUN COMPLETE!")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()