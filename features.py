"""
Feature extraction for protein sequences.
- Amino Acid Composition (AAC) — 20D
- Dipeptide Composition (DPC) — 400D
- Conjoint Triad (CT) — 343D
- Combined (AAC+DPC+CT) — 763D
- Integer encoding for CNN/BiLSTM
"""
import numpy as np
from itertools import product
from collections import Counter
from tqdm import tqdm
from config import STANDARD_AAS


def amino_acid_composition(sequences):
    """Amino Acid Composition — frequency of each of 20 amino acids. Output: (N, 20)."""
    features = []
    for seq in tqdm(sequences, desc="AAC"):
        total = max(len(seq), 1)
        counts = Counter(seq)
        features.append([counts.get(aa, 0) / total for aa in STANDARD_AAS])
    return np.array(features, dtype=np.float32)


def dipeptide_composition(sequences):
    """Dipeptide Composition — frequency of all 400 amino acid pairs. Output: (N, 400)."""
    dipeptides = [''.join(dp) for dp in product(STANDARD_AAS, repeat=2)]
    features = []
    for seq in tqdm(sequences, desc="DPC"):
        total = max(len(seq) - 1, 1)
        counts = Counter(seq[i:i+2] for i in range(len(seq) - 1))
        features.append([counts.get(dp, 0) / total for dp in dipeptides])
    return np.array(features, dtype=np.float32)


def conjoint_triad(sequences):
    """
    Conjoint Triad — maps 20 AAs to 7 physicochemical classes,
    then counts all 7x7x7 = 343 triad frequencies. Output: (N, 343).
    """
    # Amino acid grouping by physicochemical properties (Shen et al., 2007)
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    aa_class = {}
    for i, group in enumerate(groups):
        for aa in group:
            aa_class[aa] = i

    features = []
    for seq in tqdm(sequences, desc="CT"):
        classes = [aa_class.get(aa, -1) for aa in seq]
        classes = [c for c in classes if c >= 0]  # Remove unknown

        triad_counts = np.zeros(343, dtype=np.float32)
        for i in range(len(classes) - 2):
            idx = classes[i] * 49 + classes[i+1] * 7 + classes[i+2]
            triad_counts[idx] += 1

        total = max(triad_counts.sum(), 1)
        features.append(triad_counts / total)

    return np.array(features, dtype=np.float32)


def encode_sequences(sequences, max_len=1000):
    """
    Integer-encode sequences for CNN/BiLSTM.
    0 = padding, 1-20 = amino acids.
    Returns: encoded (N, max_len) int64, lengths (N,) int64
    """
    vocab = {aa: i + 1 for i, aa in enumerate(STANDARD_AAS)}  # 0 reserved for padding
    encoded = []
    lengths = []
    for seq in sequences:
        enc = [vocab.get(aa, 0) for aa in seq[:max_len]]
        lengths.append(len(enc))
        enc += [0] * (max_len - len(enc))  # Pad to max_len
        encoded.append(enc)
    return np.array(encoded, dtype=np.int64), np.array(lengths, dtype=np.int64)


def extract_all_classical(sequences):
    """Extract all classical features. Returns dict of feature arrays."""
    aac = amino_acid_composition(sequences)
    dpc = dipeptide_composition(sequences)
    ct = conjoint_triad(sequences)
    combined = np.hstack([aac, dpc, ct])

    print(f"\nFeature dimensions:")
    print(f"  AAC:      {aac.shape}")
    print(f"  DPC:      {dpc.shape}")
    print(f"  CT:       {ct.shape}")
    print(f"  Combined: {combined.shape}")

    return {'aac': aac, 'dpc': dpc, 'ct': ct, 'combined': combined}