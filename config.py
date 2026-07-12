"""
Configuration for Protein Subcellular Localization Benchmarking.
All paths, hyperparameters, and model settings in one place.
"""
from pathlib import Path
import torch

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "saved_models"

# Create all directories
for _d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATASET
# ============================================================
STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
RANDOM_SEED = 42

# ============================================================
# TRAINING
# ============================================================
MAX_SEQ_LEN = 1000        # Truncate/pad sequences for CNN/BiLSTM
BATCH_SIZE = 32
NUM_EPOCHS = 50
PATIENCE = 10              # Early stopping patience
LR = 1e-3
CV_FOLDS = 5

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ESM-2 Settings (650M model, fits in 20GB VRAM)
# ============================================================
ESM_MODEL = "esm2_t33_650M_UR50D"
ESM_LAYER = 33             # Last layer
ESM_DIM = 1280             # Embedding dimension
ESM_BATCH = 4              # Batch size for extraction

# ============================================================
# ProtT5 Settings (half-precision, fits in 20GB VRAM)
# ============================================================
PROTT5_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
PROTT5_TOKENIZER = "Rostlab/prot_t5_xl_uniref50"
PROTT5_DIM = 1024          # Embedding dimension
PROTT5_BATCH = 2           # Small batch due to model size
PROTT5_MAX_LEN = 1024      # Max tokens