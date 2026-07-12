"""Training script for MS-LANet using Focal Loss and early stopping."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- IMPORT SKLEARN AND NUMPY FIRST TO PREVENT WINDOWS DLL COLLISION ---
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import numpy as np

# --- NOW IMPORT PYTORCH ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
import sys

from config import DEVICE, MODELS_DIR, NUM_EPOCHS, PATIENCE
from data_loader_residue import get_residue_loaders
from ms_lanet import MSLANet

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_class_weights(labels, num_classes=10):
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * class_counts.clip(min=1))
    return torch.FloatTensor(weights).to(DEVICE)

def train_ms_lanet():
    print(f"Using Device: {DEVICE}")
    
    # 1. Load Data
    print("Loading residue-level data loaders...")
    train_loader, val_loader, test_loader, y_train = get_residue_loaders(batch_size=32)
    
    # 2. Initialize Model
    model = MSLANet(embedding_dim=1280, num_classes=10).to(DEVICE)
    print("MS-LANet Model Initialized.")
    
    # 3. Setup Loss and Optimizer
    alpha = compute_class_weights(y_train, num_classes=10)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    # --- HYPERPARAMETER ADJUSTMENTS FOR REGULARIZATION ---
    tuned_lr = 5e-4        # Lower learning rate (down from 1e-3)
    tuned_weight_decay = 1e-3  # Stronger L2 regularization (up from 1e-4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=tuned_lr, weight_decay=tuned_weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True
    )
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "ms_lanet_best.pt"
    
    best_val_f1 = 0
    epochs_no_improve = 0
    
    print("\nStarting Training Loop (Tuned Reg)...")
    print("=" * 70)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        start_time = time.time()
        
        for embeddings, mask, labels in train_loader:
            embeddings, mask, labels = embeddings.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            
            logits = model(embeddings, mask)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
        avg_train_loss = train_loss / train_batches
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, mask, labels in val_loader:
                embeddings, mask, labels = embeddings.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
                
                logits = model(embeddings, mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_batches += 1
                
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / val_batches
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_mcc = matthews_corrcoef(all_labels, all_preds)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val MCC: {val_mcc:.4f} | ({epoch_time:.1f}s)")
        
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 Best Model Updated & Saved (Val Macro-F1: {best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered. No validation improvement for {PATIENCE} epochs.")
                break
                
    print(f"\nTraining Complete! Best Validation Macro-F1: {best_val_f1:.4f}")
    print(f"Model saved to: {best_model_path}")

if __name__ == "__main__":
    train_ms_lanet()