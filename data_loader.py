"""
Load and preprocess DeepLoc protein subcellular localization data.
Supports: FASTA files, CSV files, HuggingFace datasets.
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import RAW_DIR, PROCESSED_DIR, RANDOM_SEED


def parse_fasta(fasta_path):
    """
    Parse FASTA file with DeepLoc-style headers.
    Expected format:  >ACCESSION LOCATION [train/test]
    Location names may contain underscores (Cell_membrane) or spaces.
    """
    records = []
    acc, seq_parts, loc, split_type = None, [], None, None

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # Save previous record
                if acc is not None:
                    records.append({
                        'accession': acc,
                        'sequence': ''.join(seq_parts),
                        'location': loc,
                        'split': split_type
                    })
                seq_parts = []

                # Parse header
                parts = line[1:].split()
                acc = parts[0]

                # Check if last word is train/test
                remaining = parts[1:]
                if remaining and remaining[-1].lower() in ('train', 'test', 'training', 'testing', 'validation'):
                    split_type = 'train' if 'train' in remaining[-1].lower() else 'test'
                    remaining = remaining[:-1]
                else:
                    split_type = None

                loc = '_'.join(remaining) if remaining else 'Unknown'
            else:
                seq_parts.append(line.upper())

    # Don't forget the last record
    if acc is not None:
        records.append({
            'accession': acc,
            'sequence': ''.join(seq_parts),
            'location': loc,
            'split': split_type
        })

    df = pd.DataFrame(records)
    df['length'] = df['sequence'].str.len()
    return df


def load_from_csv(csv_path):
    """Load data from CSV, auto-detect column names."""
    df = pd.read_csv(csv_path)
    # Try to standardize column names
    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ['accession', 'acc', 'protein_id', 'id']:
            rename_map[col] = 'accession'
        elif cl in ['sequence', 'seq']:
            rename_map[col] = 'sequence'
        elif cl in ['location', 'localization', 'label', 'class']:
            rename_map[col] = 'location'
        elif cl in ['partition', 'split', 'fold']:
            rename_map[col] = 'split'

    df = df.rename(columns=rename_map)
    if 'length' not in df.columns and 'sequence' in df.columns:
        df['length'] = df['sequence'].str.len()
    return df


def load_from_huggingface():
    """Download DeepLoc dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print("  Downloading from HuggingFace (bloyal/deeploc)...")
        ds = load_dataset("bloyal/deeploc")

        dfs = []
        for split_name in ds:
            split_df = ds[split_name].to_pandas()
            split_df['split'] = 'train' if 'train' in split_name else 'test'
            dfs.append(split_df)

        df = pd.concat(dfs, ignore_index=True)
        print(f"  Downloaded {len(df)} sequences")

        # Standardize basic columns using exact matches
        rename_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ['sequence', 'seq']:
                rename_map[col] = 'sequence'
            elif cl in ['acc', 'id', 'accession']:
                rename_map[col] = 'accession'
        df = df.rename(columns=rename_map)

        if 'accession' not in df.columns:
            df['accession'] = [f'prot_{i}' for i in range(len(df))]
        if 'length' not in df.columns and 'sequence' in df.columns:
            df['length'] = df['sequence'].str.len()

        return df
    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        return None


def load_data():
    """Auto-detect and load DeepLoc data from any available source."""
    print("\nSearching for data...")
    df = None

    # 1. Try local FASTA files
    for f in sorted(RAW_DIR.glob('*.fasta')) + sorted(RAW_DIR.glob('*.fa')) + sorted(RAW_DIR.glob('*.txt')):
        print(f"  Found FASTA: {f}")
        df = parse_fasta(f)
        break

    # 2. Try local CSV files
    if df is None:
        for f in sorted(RAW_DIR.glob('*.csv')) + sorted(RAW_DIR.glob('*.tsv')):
            # Make sure we don't load a corrupted file that was saved earlier
            if 'deeploc_huggingface' in f.name:
                continue
            print(f"  Found CSV: {f}")
            df = load_from_csv(f)
            break

    # 3. Try HuggingFace
    if df is None:
        print("  No local files found. Trying HuggingFace...")
        df = load_from_huggingface()
        if df is not None:
            # Save locally for future use
            df.to_csv(RAW_DIR / "deeploc_huggingface.csv", index=False)

    # 4. Nothing worked
    if df is None:
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"ERROR: Could not find or download DeepLoc dataset!\n"
            f"Please download manually:\n"
            f"  1. Go to: https://services.healthtech.dtu.dk/services/DeepLoc-2.1/\n"
            f"  2. Download the data files\n"
            f"  3. Place them in: {RAW_DIR}\n"
            f"{'='*60}"
        )

    # 5. Check and handle multi-label columns if present
    loc_cols = [
        'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane', 
        'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 
        'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome'
    ]
    existing_loc_cols = [c for c in loc_cols if c in df.columns]
    
    if len(existing_loc_cols) > 0 and 'location' not in df.columns:
        print("  Multi-label columns detected. Converting to a 10-class single-label task...")
        # Keep only proteins that belong to exactly one location
        loc_sums = df[existing_loc_cols].sum(axis=1)
        df = df[loc_sums == 1].copy()
        # Assign the single active class as the 'location'
        df['location'] = df[existing_loc_cols].idxmax(axis=1)
        print(f"  Filtered out multi-localized proteins: {len(df)} sequences remaining.")

    return df


def preprocess(df, min_len=10, max_len=6000):
    """Clean sequences and encode labels."""
    n_before = len(df)

    # Ensure required columns exist
    assert 'sequence' in df.columns, f"Missing 'sequence' column. Available: {list(df.columns)}"
    assert 'location' in df.columns, f"Missing 'location' column. Available: {list(df.columns)}"

    # Clean sequences: keep only standard amino acids
    df = df.copy()
    df['sequence'] = df['sequence'].astype(str).str.upper()
    df['sequence'] = df['sequence'].apply(lambda s: re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', s))
    df['length'] = df['sequence'].str.len()

    # Filter by length
    df = df[(df['length'] >= min_len) & (df['length'] <= max_len)]

    # Drop duplicates
    df = df.drop_duplicates(subset='sequence').reset_index(drop=True)

    print(f"\nPreprocessing: {n_before} → {len(df)} sequences")

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['location'])

    print(f"\nClasses ({len(le.classes_)}):")
    for i, cls_name in enumerate(le.classes_):
        count = (df['label'] == i).sum()
        print(f"  [{i}] {cls_name}: {count} samples")

    return df, le


def split_data(df):
    """Get train/test split — use provided split if available, else stratified random."""
    if 'split' in df.columns and df['split'].notna().any():
        train_df = df[df['split'].str.lower().str.contains('train', na=False)].copy()
        test_df = df[df['split'].str.lower().str.contains('test', na=False)].copy()
        if len(train_df) > 100 and len(test_df) > 100:
            print(f"\nUsing provided split: {len(train_df)} train / {len(test_df)} test")
            return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # Fallback: stratified random split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=RANDOM_SEED
    )
    print(f"\nStratified random split: {len(train_df)} train / {len(test_df)} test")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)