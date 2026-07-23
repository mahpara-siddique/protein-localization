# MS-LANet v2: Multi-Head Self-Attention Pooling and Consensus Ensemble for Protein Subcellular Localization

MS-LANet v2 is a novel, high-performance deep learning model designed to predict protein subcellular localization from primary sequences using pre-trained Protein Language Model (pLM) embeddings. 

Unlike standard approaches that rely on global mean-pooling (which washes out localized targeting signals) or simple sliding-window convolutions, MS-LANet v2 introduces a **True Multi-Head Self-Attention Pooling** mechanism. Using a learnable query vector acting as a biological probe, the network dynamically computes query-key interactions over residue-level **ESM-2 (650M)** representations to extract position-specific sorting signals (e.g., N-terminal signal peptides, nuclear localization motifs).

Combined with global mean-pooling and blended with a Support Vector Machine (SVM RBF) classifier in a **5-Fold Cross-Validation Ensemble**, this framework achieves state-of-the-art performance on the standard eukaryotic 10-class benchmark.

---

## 🏆 Benchmark Comparison (DeepLoc Test Set)

Our final ensembled model outperforms multiple established publication baselines, including Meta AI's giant 15-billion parameter model and RostLab ProtTrans models:

| Model / Architecture | Q10 Accuracy | Matthews Correlation (MCC) | Scientific Status |
| :--- | :---: | :---: | :--- |
| **DeepLoc 1.0** (CNN + BiLSTM + PSSM) | 78.00% | 0.7200 | Published Baseline |
| **ESM-1b** (650M Mean Embedding) | 80.00% | - | Published Baseline |
| **ProtTrans (ProtT5-XL)** | 81.00% | - | Published Baseline |
| **ESM-2 (15B Parameter Model)** | 81.80% | - | Published Baseline |
| **DeepLoc 2.0 (ProtT5-XL)** | 83.00% | 0.7800 | Published Baseline |
| **Ankh Large** (Optimized Transformer) | 83.20% | - | Published Baseline |
| **Our SVM RBF Baseline (ESM-2 Mean)** | **81.29% ± 0.90%** | **0.7765 ± 0.0108** | **5-Fold CV Verified** |
| **Our Model: 5-Fold MS-LANet v2 alone** | **84.26%** | **0.8118** | **True Self-Attention** |
| **Our Model: Final Blended Ensemble** | **84.47%** | **0.8142** | **🏆 NEW SOTA BENCHMARK** |

---

## 🔬 Architectural Ablation Study

To systematically justify each design choice, we conducted a 12-configuration ablation study on the benchmark partition:

| # | Experiment Name | Pooling Strategy | Loss Function | Dropout | LayerNorm | Test Acc | Test Macro-F1 | Test MCC |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 01 | `mean_only` | Global Mean Only | Focal Loss | 0.4 | True | 78.69% | 0.7475 | 0.7473 |
| 02 | `max_only` | Global Max Only | Focal Loss | 0.4 | True | 76.04% | 0.7091 | 0.7143 |
| 03 | `mean_max` | Mean + Max | Focal Loss | 0.4 | True | 68.20% | 0.6482 | 0.6258 |
| 04 | `conv_pool_only` | Conv Weighted Pool | Focal Loss | 0.4 | True | 80.69% | 0.7625 | 0.7704 |
| 05 | `conv_pool_mean` | Conv Pool + Mean | Focal Loss | 0.4 | True | 80.13% | 0.7555 | 0.7633 |
| 06 | `self_attn_only` | True Self-Attn Only | Focal Loss | 0.4 | True | 79.55% | 0.7491 | 0.7562 |
| 07 | `self_attn_mean` | Self-Attn + Mean | Focal Loss | 0.4 | True | 80.66% | 0.7646 | 0.7688 |
| 08 | `ce_loss` | Self-Attn + Mean | Cross-Entropy | 0.4 | True | 80.13% | 0.7583 | 0.7638 |
| 09 | `label_smoothing` | Self-Attn + Mean | Label Smooth (0.1) | 0.4 | True | **81.57%** | **0.7691** | **0.7794** |
| 10 | `dropout_0.1` | Self-Attn + Mean | Focal Loss | 0.1 | True | 79.13% | 0.7585 | 0.7509 |
| 11 | `dropout_0.3` | Self-Attn + Mean | Focal Loss | 0.3 | True | 79.99% | 0.7650 | 0.7622 |
| 12 | `no_layernorm` | Self-Attn + Mean | Focal Loss | 0.4 | False | 77.90% | 0.7401 | 0.7381 |

---

## 📂 Project Structure

```text
├── config.py                      # Global paths, hyperparameters, and device configs
├── data_loader_residue.py         # Residue-level PyTorch Dataset and Stratified splits
├── ms_lanet_v2.py                 # Core MS-LANet v2 architecture (True Self-Attention)
├── extract_residue_embeddings.py  # GPU script to extract token-level ESM-2 tensors
├── run_baseline_cv.py             # 5-Fold Cross-Validation for 9 baseline classifiers
├── run_ablation.py                # Automated runner for 12 ablation study experiments
├── train_ms_lanet_v2_cv.py        # 5-Fold CV training loop for optimal MS-LANet v2
├── evaluate_ensemble_cv.py        # Final evaluation & ensembling (MS-LANet v2 + SVM)
├── plot_paper_figures.py          # Script to generate publication charts
├── LICENSE                        # Strict proprietary license
└── README.md                      # Project documentation

🚀 Reproduction Steps
1. Extract ESM-2 Residue Tensors
python extract_residue_embeddings.py

2. Run 5-Fold Baseline Cross-Validation
python run_baseline_cv.py

3. Run Architectural Ablation Study
python run_ablation.py

4. Train Optimal 5-Fold MS-LANet v2 Model
python train_ms_lanet_v2_cv.py

5. Evaluate Final Consensus Ensemble
python evaluate_ensemble_cv.py
