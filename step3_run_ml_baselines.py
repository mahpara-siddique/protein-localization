"""
STEP 3: Run the 6 classical ML baselines.
Trains each model on:
  - Handcrafted features (AAC+DPC+CT)
  - ESM-2 Embeddings
Saves all scores in a results CSV.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config import FEATURES_DIR, PROCESSED_DIR, RESULTS_DIR
from trainer import train_sklearn_model
from evaluation import evaluate_predictions, plot_confusion_matrix


def load_dataset_features(feature_type):
    """Load train/test split for a specific feature set."""
    X_train = np.load(FEATURES_DIR / f"train_{feature_type}.npy")
    X_test = np.load(FEATURES_DIR / f"test_{feature_type}.npy")
    y_train = np.load(FEATURES_DIR / "train_labels.npy")
    y_test = np.load(FEATURES_DIR / "test_labels.npy")
    return X_train, X_test, y_train, y_test


def main():
    print("=" * 60)
    print("STEP 3: Running Classical ML Baselines")
    print("=" * 60)

    # Load label encoder classes
    with open(PROCESSED_DIR / "label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)
    classes = le.classes_

    # Create placeholder for results
    results = []

    # Map baseline models to run
    models = {
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "SVM (RBF)": SVC(C=10.0, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric='mlogloss', n_jobs=-1, random_state=42)
    }

    # Run experiments
    for model_name, clf in models.items():
        # ---- Experiment A: Handcrafted Features (AAC + DPC + CT) ----
        print(f"\n{'─'*50}")
        print(f"Running {model_name} on Handcrafted Features...")
        X_train, X_test, y_train, y_test = load_dataset_features("combined")
        
        # Scale features for distance/margin based classifiers
        if model_name in ["KNN", "Logistic Regression", "SVM (RBF)"]:
            pipeline = make_pipeline(StandardScaler(), clf)
        else:
            pipeline = clf
            
        preds, _ = train_sklearn_model(pipeline, X_train, y_train, X_test)
        metrics = evaluate_predictions(y_test, preds, classes, f"{model_name} (Handcrafted)")
        
        results.append({
            'Model': model_name,
            'Features': 'AAC+DPC+CT (763D)',
            'Accuracy': metrics['accuracy'],
            'Macro-F1': metrics['macro_f1'],
            'MCC': metrics['mcc']
        })

        # ---- Experiment B: ESM-2 Embeddings ----
        print(f"\nRunning {model_name} on ESM-2 Embeddings...")
        X_train, X_test, y_train, y_test = load_dataset_features("esm2")
        
        if model_name in ["KNN", "Logistic Regression", "SVM (RBF)"]:
            pipeline = make_pipeline(StandardScaler(), clf)
        else:
            pipeline = clf
            
        preds, _ = train_sklearn_model(pipeline, X_train, y_train, X_test)
        metrics = evaluate_predictions(y_test, preds, classes, f"{model_name} (ESM-2)")
        
        # Plot confusion matrix for SVM-ESM2 (usually top ML baseline)
        if model_name == "SVM (RBF)":
            plot_confusion_matrix(y_test, preds, classes, "svm_esm2_confusion_matrix.png")

        results.append({
            'Model': model_name,
            'Features': 'ESM-2 (1280D)',
            'Accuracy': metrics['accuracy'],
            'Macro-F1': metrics['macro_f1'],
            'MCC': metrics['mcc']
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "ml_baselines_results.csv", index=False)
    print(f"\n✅ ML Baselines Complete! Saved: {RESULTS_DIR / 'ml_baselines_results.csv'}")


if __name__ == "__main__":
    main()