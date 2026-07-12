"""
STEP 1: Load data → EDA → Extract classical features → Encode sequences.
Run this first. No GPU needed for this step.
Expected time: ~5 minutes.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from config import PROCESSED_DIR, FEATURES_DIR, FIGURES_DIR, MAX_SEQ_LEN
from data_loader import load_data, preprocess, split_data
from features import extract_all_classical, encode_sequences


def main():
    print("=" * 60)
    print("STEP 1: Data Preparation & Feature Extraction")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------
    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"  Loaded: {len(df)} sequences")
    print(f"  Columns: {list(df.columns)}")

    # --------------------------------------------------------
    # 2. PREPROCESS
    # --------------------------------------------------------
    print("\n[2/5] Preprocessing...")
    df, le = preprocess(df)

    # Save processed data
    df.to_csv(PROCESSED_DIR / "processed_data.csv", index=False)
    with open(PROCESSED_DIR / "label_encoder.pkl", 'wb') as f:
        pickle.dump(le, f)
    print(f"  Saved processed data → {PROCESSED_DIR}")

    # --------------------------------------------------------
    # 3. EDA VISUALIZATIONS
    # --------------------------------------------------------
    print("\n[3/5] Generating EDA plots...")

    # Class distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    counts = df['location'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor='gray')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=10)
    ax.set_title('Class Distribution — DeepLoc Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples')
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print("  ✓ Saved: class_distribution.png")

    # Sequence length distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['length'], bins=60, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title('Sequence Length Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sequence Length (amino acids)')
    ax.set_ylabel('Count')
    median_len = df['length'].median()
    ax.axvline(median_len, color='red', linestyle='--', linewidth=1.5,
               label=f'Median: {median_len:.0f}')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "length_distribution.png", dpi=150)
    plt.close()
    print("  ✓ Saved: length_distribution.png")

    # Summary
    print(f"\n  📊 Dataset Summary:")
    print(f"     Total sequences:    {len(df)}")
    print(f"     Number of classes:  {df['location'].nunique()}")
    print(f"     Seq lengths:        min={df['length'].min()}, max={df['length'].max()}, median={median_len:.0f}")
    print(f"     Imbalance ratio:    {counts.max()}/{counts.min()} = {counts.max()/counts.min():.1f}x")

    # --------------------------------------------------------
    # 4. TRAIN / TEST SPLIT
    # --------------------------------------------------------
    print("\n[4/5] Splitting data...")
    train_df, test_df = split_data(df)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"  Saved: train.csv ({len(train_df)}), test.csv ({len(test_df)})")

    # --------------------------------------------------------
    # 5. EXTRACT CLASSICAL FEATURES
    # --------------------------------------------------------
    print("\n[5/5] Extracting classical features...")

    # Train features
    print("\n  Training set features:")
    train_feats = extract_all_classical(train_df['sequence'].tolist())
    for name, feat in train_feats.items():
        path = FEATURES_DIR / f"train_{name}.npy"
        np.save(path, feat)
        print(f"    ✓ train_{name}.npy  →  {feat.shape}")

    # Test features
    print("\n  Test set features:")
    test_feats = extract_all_classical(test_df['sequence'].tolist())
    for name, feat in test_feats.items():
        path = FEATURES_DIR / f"test_{name}.npy"
        np.save(path, feat)
        print(f"    ✓ test_{name}.npy  →  {feat.shape}")

    # Encode sequences for CNN/BiLSTM
    print("\n  Encoding sequences for CNN/BiLSTM...")
    train_enc, train_lens = encode_sequences(train_df['sequence'].tolist(), MAX_SEQ_LEN)
    test_enc, test_lens = encode_sequences(test_df['sequence'].tolist(), MAX_SEQ_LEN)

    np.save(FEATURES_DIR / "train_encoded.npy", train_enc)
    np.save(FEATURES_DIR / "train_lengths.npy", train_lens)
    np.save(FEATURES_DIR / "test_encoded.npy", test_enc)
    np.save(FEATURES_DIR / "test_lengths.npy", test_lens)
    print(f"    ✓ Encoded sequences: train {train_enc.shape}, test {test_enc.shape}")

    # Save labels separately for easy loading
    np.save(FEATURES_DIR / "train_labels.npy", train_df['label'].values)
    np.save(FEATURES_DIR / "test_labels.npy", test_df['label'].values)
    print(f"    ✓ Labels saved")

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("✅ STEP 1 COMPLETE!")
    print(f"   Processed data:    {PROCESSED_DIR}")
    print(f"   Cached features:   {FEATURES_DIR}")
    print(f"   EDA figures:       {FIGURES_DIR}")
    print(f"\n   Next: Run step2_extract_embeddings.py (requires GPU)")
    print("=" * 60)


if __name__ == "__main__":
    main()