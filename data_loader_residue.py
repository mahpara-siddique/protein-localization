"""Residue-level PyTorch data loader for MS-LANet."""
# --- IMPORT SKLEARN AND NUMPY FIRST ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# --- NOW IMPORT PYTORCH ---
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

from config import PROCESSED_DIR, FEATURES_DIR, BATCH_SIZE, RANDOM_SEED, MAX_SEQ_LEN

# ... (the rest of your data_loader_residue.py code remains exactly the same)

class ResidueEmbeddingDataset(Dataset):
    def __init__(self, accessions: list, labels: list, embedding_dir: Path, max_len: int = MAX_SEQ_LEN):
        self.accessions = accessions
        self.labels = torch.LongTensor(labels)
        self.embedding_dir = Path(embedding_dir)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.accessions)
        
    def __getitem__(self, idx):
        acc = self.accessions[idx]
        label = self.labels[idx]
        
        # Load the saved tensor [seq_len, 1280]
        emb_path = self.embedding_dir / f"{acc}.pt"
        emb = torch.load(emb_path, map_location="cpu")
        
        # Truncate if sequence exceeds max length limit
        if emb.size(0) > self.max_len:
            emb = emb[:self.max_len]
            
        seq_len = emb.size(0)
        
        # Create zero-padded embedding tensor: [max_len, 1280]
        padded_emb = torch.zeros(self.max_len, emb.size(1))
        padded_emb[:seq_len] = emb
        
        # Create boolean mask: [max_len] (True for real residues, False for padding)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:seq_len] = True
        
        return padded_emb, mask, label

def get_residue_loaders(batch_size: int = BATCH_SIZE) -> tuple:
    """Splits data and returns Train, Val, and Test PyTorch DataLoaders."""
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"train.csv or test.csv not found in {PROCESSED_DIR}")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Load Label Encoder to transform location text labels to integer IDs
    le = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
    df_train['label'] = le.transform(df_train['location'])
    df_test['label'] = le.transform(df_test['location'])
    
    # Split train.csv into train (85%) and validation (15%) splits
    df_tr, df_val = train_test_split(
        df_train, test_size=0.15, random_state=RANDOM_SEED, stratify=df_train['label']
    )
    
    embedding_dir = FEATURES_DIR / "esm2_residue"
    
    train_ds = ResidueEmbeddingDataset(df_tr['accession'].tolist(), df_tr['label'].tolist(), embedding_dir)
    val_ds = ResidueEmbeddingDataset(df_val['accession'].tolist(), df_val['label'].tolist(), embedding_dir)
    test_ds = ResidueEmbeddingDataset(df_test['accession'].tolist(), df_test['label'].tolist(), embedding_dir)
    
    # pin_memory=True speeds up transfers from RAM to GPU VRAM
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader, df_tr['label'].values

if __name__ == "__main__":
    # Test loading
    try:
        train_loader, val_loader, test_loader, _ = get_residue_loaders(batch_size=2)
        padded_emb, mask, label = next(iter(train_loader))
        print("Padded Embedding Batch Shape:", padded_emb.shape)  # Expect [2, 1000, 1280]
        print("Mask Batch Shape:", mask.shape)                    # Expect [2, 1000]
        print("Label Batch Shape:", label.shape)                  # Expect [2]
        print("Data Loader looks correct!")
    except Exception as e:
        print("Setup error or embeddings are still extracting:", e)