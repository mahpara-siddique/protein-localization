"""
STEP 4: Run the 3 Deep Learning baselines.
- MLP trained on ESM-2 embeddings
- 1D-CNN trained on integer-encoded sequences
- BiLSTM + Attention trained on integer-encoded sequences
"""
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import TensorDataset

from config import FEATURES_DIR, PROCESSED_DIR, RESULTS_DIR
from models import ProteinMLP, ProteinCNN, ProteinBiLSTM
from trainer import train_pytorch_model, predict_pytorch_model
from evaluation import evaluate_predictions, plot_confusion_matrix


def main():
    print("=" * 60)
    print("STEP 4: Running Deep Learning Baselines")
    print("=" * 60)

    # Load labels and splits
    with open(PROCESSED_DIR / "label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)
    classes = le.classes_
    num_classes = len(classes)

    y_train = torch.tensor(np.load(FEATURES_DIR / "train_labels.npy"), dtype=torch.long)
    y_test = torch.tensor(np.load(FEATURES_DIR / "test_labels.npy"), dtype=torch.long)

    results = []

    # ========================================================
    # 1. MLP ON ESM-2 EMBEDDINGS
    # ========================================================
    print("\n[1/3] Training MLP on ESM-2 Embeddings...")
    X_train_esm = torch.tensor(np.load(FEATURES_DIR / "train_esm2.npy"), dtype=torch.float32)
    X_test_esm = torch.tensor(np.load(FEATURES_DIR / "test_esm2.npy"), dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_esm, y_train)
    test_dataset = TensorDataset(X_test_esm, y_test)
    
    model = ProteinMLP(input_dim=1280, num_classes=num_classes)
    model = train_pytorch_model(model, train_dataset, test_dataset, model_name="mlp")
    
    preds = predict_pytorch_model(model, test_dataset)
    metrics = evaluate_predictions(y_test.numpy(), preds, classes, "MLP (ESM-2)")
    
    results.append({
        'Model': 'MLP',
        'Features': 'ESM-2 (1280D)',
        'Accuracy': metrics['accuracy'],
        'Macro-F1': metrics['macro_f1'],
        'MCC': metrics['mcc']
    })

    # ========================================================
    # 2. CNN ON RAW SEQUENCES
    # ========================================================
    print("\n[2/3] Training 1D-CNN on Raw sequences...")
    X_train_seq = torch.tensor(np.load(FEATURES_DIR / "train_encoded.npy"), dtype=torch.long)
    X_test_seq = torch.tensor(np.load(FEATURES_DIR / "test_encoded.npy"), dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_seq, y_train)
    test_dataset = TensorDataset(X_test_seq, y_test)
    
    model = ProteinCNN(vocab_size=21, num_classes=num_classes)
    model = train_pytorch_model(model, train_dataset, test_dataset, model_name="cnn")
    
    preds = predict_pytorch_model(model, test_dataset)
    metrics = evaluate_predictions(y_test.numpy(), preds, classes, "1D-CNN (Raw Sequence)")
    
    results.append({
        'Model': '1D-CNN',
        'Features': 'Raw Sequence (Integer Enc)',
        'Accuracy': metrics['accuracy'],
        'Macro-F1': metrics['macro_f1'],
        'MCC': metrics['mcc']
    })

    # ========================================================
    # 3. BiLSTM + ATTENTION
    # ========================================================
    print("\n[3/3] Training BiLSTM + Attention on Raw sequences...")
    train_lens = torch.tensor(np.load(FEATURES_DIR / "train_lengths.npy"), dtype=torch.long)
    test_lens = torch.tensor(np.load(FEATURES_DIR / "test_lengths.npy"), dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_seq, train_lens, y_train)
    test_dataset = TensorDataset(X_test_seq, test_lens, y_test)
    
    model = ProteinBiLSTM(vocab_size=21, num_classes=num_classes)
    model = train_pytorch_model(model, train_dataset, test_dataset, model_name="bilstm")
    
    preds = predict_pytorch_model(model, test_dataset)
    metrics = evaluate_predictions(y_test.numpy(), preds, classes, "BiLSTM (Raw Sequence)")
    
    plot_confusion_matrix(y_test.numpy(), preds, classes, "bilstm_confusion_matrix.png")
    
    results.append({
        'Model': 'BiLSTM + Attention',
        'Features': 'Raw Sequence (Integer Enc)',
        'Accuracy': metrics['accuracy'],
        'Macro-F1': metrics['macro_f1'],
        'MCC': metrics['mcc']
    })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "dl_baselines_results.csv", index=False)
    print(f"\n✅ DL Baselines Complete! Saved: {RESULTS_DIR / 'dl_baselines_results.csv'}")


if __name__ == "__main__":
    main()