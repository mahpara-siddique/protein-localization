# Baseline Model Benchmarking Summary

| Model               | Features                   | Accuracy   |   Macro-F1 |    MCC |
|:--------------------|:---------------------------|:-----------|-----------:|-------:|
| SVM (RBF)           | ESM-2 (1280D)              | 83.01%     |     0.789  | 0.7969 |
| XGBoost             | ESM-2 (1280D)              | 80.83%     |     0.7573 | 0.7701 |
| MLP                 | ESM-2 (1280D)              | 78.06%     |     0.7229 | 0.7419 |
| Logistic Regression | ESM-2 (1280D)              | 77.39%     |     0.7244 | 0.7327 |
| Random Forest       | ESM-2 (1280D)              | 76.02%     |     0.6937 | 0.7112 |
| KNN                 | ESM-2 (1280D)              | 75.58%     |     0.7022 | 0.707  |
| BiLSTM + Attention  | Raw Sequence (Integer Enc) | 62.47%     |     0.5439 | 0.5603 |
| SVM (RBF)           | AAC+DPC+CT (763D)          | 60.72%     |     0.5128 | 0.5276 |
| XGBoost             | AAC+DPC+CT (763D)          | 60.05%     |     0.4572 | 0.5139 |
| Naive Bayes         | ESM-2 (1280D)              | 53.60%     |     0.4628 | 0.4645 |
| Random Forest       | AAC+DPC+CT (763D)          | 51.56%     |     0.3495 | 0.4066 |
| Logistic Regression | AAC+DPC+CT (763D)          | 45.87%     |     0.3834 | 0.379  |
| KNN                 | AAC+DPC+CT (763D)          | 43.20%     |     0.3115 | 0.3137 |
| 1D-CNN              | Raw Sequence (Integer Enc) | 35.21%     |     0.2523 | 0.2679 |
| Naive Bayes         | AAC+DPC+CT (763D)          | 27.97%     |     0.2475 | 0.2294 |

**Best Baseline**: SVM (RBF) using ESM-2 (1280D) (MCC: 0.7969)