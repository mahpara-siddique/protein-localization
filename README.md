# MS-LANet: Multi-Scale Light Attention Network for Protein Subcellular Localization

<img width="1080" height="1080" alt="graphical_abstract" src="https://github.com/user-attachments/assets/39728915-1cd5-4bda-a18d-79d4c1789163" />


MS-LANet is a novel, high-performance deep learning model designed to predict protein subcellular localization from primary sequences using pre-trained Protein Language Model (pLM) embeddings. 

By utilizing **Light Attention Pooling** combined with **Global Mean-Pooling** over residue-level **ESM-2 (650M)** representations, MS-LANet extracts localized sorting signals (like N-terminal signal peptides or C-terminal tail motifs) alongside global sequence characteristics.

Combined with an **SVM (RBF)** classifier via a **5-Fold Cross-Validation Ensemble**, this framework achieves state-of-the-art performance on the standard eukaryotic 10-class subcellular localization benchmark.

---

## 🏆 Benchmark Results (DeepLoc Test Set)

Our final ensembled model outperforms multiple established publication baselines, including Meta AI's giant 15-billion parameter model and the RostLab ProtTrans models:

| Model / Architecture | Q10 Accuracy | Matthews Correlation (MCC) |
| :--- | :---: | :---: |
| **DeepLoc 1.0** (CNN + BiLSTM + PSSM) | 78.00% | 0.7200 |
| **ESM-1b** (650M Mean Embedding) | 80.00% | - |
| **ProtTrans (ProtT5-XL)** | 81.00% | - |
| **ESM-2 (15B Parameter Model)** | 81.80% | - |
| **DeepLoc 2.0 (ProtT5-XL)** | 83.00% | 0.7800 |
| **Ankh Large** (Optimized Transformer) | 83.20% | - |
| **Our Model: 5-Fold MS-LANet alone** | **83.40%** | **0.8011** |
| **Our Model: 5-Fold CV Ensemble (LANet + SVM)** | **83.96%** | **0.8085** |

---

## 🧬 Key Features
* **Attention-Based Pooling**: Replaces standard mean/max-pooling with a 1D-convolutional Light Attention mechanism to locate and weight sequence sorting signals.
* **5-Fold Cross-Validation Committee**: Ensembles predictions across 5 independent fold models to reduce variance and boost generalizability.
* **Focal Loss Regularization**: Mitigates severe class imbalance (e.g. Golgi apparatus, Peroxisomes) by focusing gradients on hard-to-classify minority classes.
* **Low VRAM footprint**: Runs efficiently on a single local GPU (e.g., NVIDIA RTX A4500) using cached individual residue tensors.

---

## 📂 Project Structure

```text
├── config.py                     # Global paths, hyperparameters, and device configs
├── data_loader_residue.py        # Residue-level PyTorch Dataset and Stratified splits
├── ms_lanet.py                   # MS-LANet PyTorch architecture (Attention + Mean Pool)
├── extract_residue_embeddings.py # GPU script to extract token-level ESM-2 tensors
├── train_ms_lanet_cv.py          # 5-Fold Cross-Validation training loop with Focal Loss
├── evaluate_ensemble_cv.py       # Validation and ensembling script (LANet Folds + SVM)
├── LICENSE                       # Strict proprietary license
└── README.md                     # Project documentation

🚀 How to Run the Pipeline

1. Extract Residue Tensors
Run the extraction script on your GPU machine to cache ESM-2 residue-level tensors for all sequences:
python extract_residue_embeddings.py

2. Run 5-Fold CV Training
Train the 5 separate folds of the MSLANet model using Focal Loss and early stopping:
python train_ms_lanet_cv.py

3. Evaluate the Ensemble
Load the pre-trained SVM model and blend its predictions with the 5-Fold MS-LANet ensemble (using a 0.7 / 0.3 weight combination):
python evaluate_ensemble_cv.py
