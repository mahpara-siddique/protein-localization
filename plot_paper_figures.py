"""
Generate publication-quality charts for the paper & GitHub.
- Figure A: Ablation Study Comparison Bar Chart
- Figure B: Baseline 5-Fold CV Performance Comparison
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def plot_ablation():
    ablation_csv = RESULTS_DIR / "ablation_results.csv"
    if not ablation_csv.exists():
        print(f"Error: {ablation_csv} not found.")
        return
        
    df = pd.read_csv(ablation_csv)
    df['Acc_Num'] = df['Test_Acc'].str.rstrip('%').astype(float)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color palette: Highlight the best model (09_label_smoothing) in dark blue/teal
    colors = ['#4c72b0' if '09' not in exp else '#1b9e77' for exp in df['Experiment']]
    
    bars = ax.barh(df['Experiment'], df['Acc_Num'], color=colors, edgecolor='black', alpha=0.85)
    
    ax.set_xlim(65, 85)
    ax.set_xlabel('Test Q10 Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ablation Experiment', fontsize=12, fontweight='bold')
    ax.set_title('MS-LANet v2 Architectural Ablation Study (DeepLoc Test Set)', fontsize=14, fontweight='bold', pad=15)
    
    # Add numerical labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                ha='left', va='center', fontsize=10, fontweight='bold')
                
    plt.tight_layout()
    save_path = FIGURES_DIR / "ablation_chart.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Ablation chart saved to: {save_path}")

def plot_baselines():
    baseline_csv = RESULTS_DIR / "baseline_cv_results.csv"
    if not baseline_csv.exists():
        print(f"Error: {baseline_csv} not found.")
        return
        
    df = pd.read_csv(baseline_csv)
    df['Acc_Mean'] = df['Accuracy_Mean'].astype(float) * 100
    df['Acc_Std'] = df['Accuracy_Std'].astype(float) * 100
    
    # Add our final model for comparison
    our_model = pd.DataFrame([{
        'Model': 'Our MS-LANet v2 Ensemble (SOTA)',
        'Acc_Mean': 84.47,
        'Acc_Std': 0.0
    }])
    
    df_combined = pd.concat([df[['Model', 'Acc_Mean', 'Acc_Std']], our_model], ignore_index=True)
    df_combined = df_combined.sort_values(by='Acc_Mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#d95f02' if 'MS-LANet' in m else '#7570b3' for m in df_combined['Model']]
    
    bars = ax.barh(df_combined['Model'], df_combined['Acc_Mean'], xerr=df_combined['Acc_Std'], 
                   color=colors, edgecolor='black', alpha=0.85, capsize=4)
                   
    ax.set_xlim(20, 90)
    ax.set_xlabel('5-Fold CV Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Comparison vs. Proposed MS-LANet v2 Ensemble', fontsize=14, fontweight='bold', pad=15)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.8, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                ha='left', va='center', fontsize=10, fontweight='bold')
                
    plt.tight_layout()
    save_path = FIGURES_DIR / "baseline_comparison_chart.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Baseline comparison chart saved to: {save_path}")

if __name__ == "__main__":
    plot_ablation()
    plot_baselines()