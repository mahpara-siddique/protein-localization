"""Script to extract per-residue ESM-2 embeddings and save them individually by accession ID."""
import numpy as np
import pandas as pd
import torch
import esm
from pathlib import Path
from tqdm import tqdm
import sys
import os

from config import PROCESSED_DIR, FEATURES_DIR, DEVICE

def main():
    print(f"Using device: {DEVICE}")
    
    # Load both train.csv and test.csv
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        print(f"Error: train.csv or test.csv not found in {PROCESSED_DIR}. Run step1_prepare_data.py first.")
        return
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Combine them to extract embeddings for all sequences
    df = pd.concat([df_train, df_test], ignore_index=True)
    sequences = df['sequence'].tolist()
    accessions = df['accession'].tolist()
    print(f"Loaded {len(sequences)} sequences in total (Train: {len(df_train)}, Test: {len(df_test)}).")

    # Create output directory
    output_dir = FEATURES_DIR / "esm2_residue"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving individual tensors to {output_dir}")

    # Load ESM-2 model (650M parameter version)
    model_name = "esm2_t33_650M_UR50D"
    print(f"Loading {model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()
    
    num_layers = 33
    batch_size = 8  # Adjust based on GPU VRAM (8 is safe for 20GB VRAM)
    
    # Loop over sequences and extract
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting Residue Tensors"):
        batch_slice = sequences[i : i + batch_size]
        batch_accs = accessions[i : i + batch_size]
        
        # ESM-2 max token length is 1024 (including BOS/EOS, so we slice sequence to 1022)
        batch_seqs = [(acc, seq[:1022]) for acc, seq in zip(batch_accs, batch_slice)]
        
        _, _, toks = batch_converter(batch_seqs)
        toks = toks.to(DEVICE)
        
        with torch.no_grad():
            results = model(toks, repr_layers=[num_layers])
            token_embs = results["representations"][num_layers]  # Shape: [B, L, 1280]
            
            for j in range(len(batch_slice)):
                acc = batch_accs[j]
                seq_len = min(len(batch_slice[j]), 1022)
                
                # Exclude BOS (index 0) and EOS (index seq_len + 1)
                residue_emb = token_embs[j, 1 : seq_len + 1, :].cpu()  # Shape: [seq_len, 1280]
                
                # Save individual tensor using accession ID
                save_path = output_dir / f"{acc}.pt"
                torch.save(residue_emb, save_path)

    print("\nResidue-level embeddings extracted and saved successfully!")

if __name__ == "__main__":
    main()