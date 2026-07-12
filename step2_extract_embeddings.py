"""
STEP 2: Extract PLM embeddings using GPU.
- ESM-2 (650M) → 1280D per protein
- ProtT5-XL (half-precision) → 1024D per protein

Requires GPU with 20GB VRAM.
Expected time: ~60-90 minutes total.
"""
import numpy as np
import pandas as pd
from config import (
    DEVICE, PROCESSED_DIR, FEATURES_DIR,
    ESM_MODEL, ESM_DIM, PROTT5_MODEL, PROTT5_DIM
)
from embeddings import extract_esm2, extract_prott5


def main():
    print("=" * 60)
    print("STEP 2: PLM Embedding Extraction")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    if str(DEVICE) == 'cpu':
        print("\n⚠️  WARNING: No GPU detected! This will be extremely slow.")
        print("   Consider using a smaller ESM-2 model (edit config.py).")
        resp = input("   Continue anyway? (y/n): ")
        if resp.lower() != 'y':
            return

    # Load sequences
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    train_seqs = train_df['sequence'].tolist()
    test_seqs = test_df['sequence'].tolist()

    total = len(train_seqs) + len(test_seqs)

    # ========================================================
    # 1. ESM-2 EMBEDDINGS
    # ========================================================
    print(f"\n[1/2] ESM-2 Embeddings ({ESM_MODEL})")
    print(f"  Total sequences: {total}")
    print(f"  Expected time: ~20-40 minutes")

    esm_train_path = FEATURES_DIR / "train_esm2.npy"
    esm_test_path = FEATURES_DIR / "test_esm2.npy"

    if not esm_train_path.exists():
        print("\n  Extracting TRAIN embeddings...")
        extract_esm2(train_seqs, save_path=esm_train_path)
    else:
        print(f"  ✓ Already cached: {esm_train_path}")

    if not esm_test_path.exists():
        print("\n  Extracting TEST embeddings...")
        extract_esm2(test_seqs, save_path=esm_test_path)
    else:
        print(f"  ✓ Already cached: {esm_test_path}")

    # Verify
    train_esm = np.load(esm_train_path)
    test_esm = np.load(esm_test_path)
    print(f"\n  ESM-2 shapes: train={train_esm.shape}, test={test_esm.shape}")

    # ========================================================
    # 2. PROTT5 EMBEDDINGS
    # ========================================================
    print(f"\n[2/2] ProtT5 Embeddings ({PROTT5_MODEL})")
    print(f"  Total sequences: {total}")
    print(f"  Expected time: ~30-60 minutes")
    print(f"  ⚠️  First run downloads ~3GB model")

    pt5_train_path = FEATURES_DIR / "train_prott5.npy"
    pt5_test_path = FEATURES_DIR / "test_prott5.npy"

    if not pt5_train_path.exists():
        print("\n  Extracting TRAIN embeddings...")
        extract_prott5(train_seqs, save_path=pt5_train_path)
    else:
        print(f"  ✓ Already cached: {pt5_train_path}")

    if not pt5_test_path.exists():
        print("\n  Extracting TEST embeddings...")
        extract_prott5(test_seqs, save_path=pt5_test_path)
    else:
        print(f"  ✓ Already cached: {pt5_test_path}")

    # Verify
    train_pt5 = np.load(pt5_train_path)
    test_pt5 = np.load(pt5_test_path)
    print(f"\n  ProtT5 shapes: train={train_pt5.shape}, test={test_pt5.shape}")

    # ========================================================
    # SUMMARY
    # ========================================================
    print("\n" + "=" * 60)
    print("✅ STEP 2 COMPLETE!")
    print(f"\n  All cached features in {FEATURES_DIR}:")
    for f in sorted(FEATURES_DIR.glob("*.npy")):
        arr = np.load(f)
        print(f"    {f.name:30s}  shape={arr.shape}")
    print(f"\n  Next: Run step3_run_baselines.py")
    print("=" * 60)


if __name__ == "__main__":
    main()