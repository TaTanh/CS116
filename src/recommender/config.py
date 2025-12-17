"""Configuration file for the recommender system."""

from pathlib import Path

# ========== Data Paths ==========
# Default data directory
DATA_DIR = Path(r"E:\Nam_3_HK1\PythonMayHoc\dataset")

# Default file paths (using glob patterns for chunked files)
TRANSACTIONS_PATTERN = str(DATA_DIR / "sales_pers.purchase_history_daily_chunk_*.parquet")
ITEMS_PATTERN = str(DATA_DIR / "sales_pers.item_chunk_*.parquet")
USERS_PATTERN = str(DATA_DIR / "sales_pers.user_chunk_*.parquet")

# Legacy single file paths (for backward compatibility)
TRANSACTIONS_PATH = TRANSACTIONS_PATTERN
ITEMS_PATH = ITEMS_PATTERN
USERS_PATH = USERS_PATTERN

# ========== Output Paths ==========
OUTPUT_DIR = Path("outputs")
MODELS_DIR = OUTPUT_DIR / "models"
FEATURES_DIR = OUTPUT_DIR / "features"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
FEATURES_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)

# ========== Model Parameters ==========
DEFAULT_MODEL_TYPE = "logistic"  # or "lightgbm"
RANDOM_STATE = 42

# ========== Feature Engineering Parameters ==========
# Time windows (can be overridden)
DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_PREDICTION_DAYS = 30

# Candidate generation
MAX_CANDIDATES_PER_USER = 200
TOP_K_POPULAR_ITEMS = 50

# ========== Evaluation Parameters ==========
DEFAULT_K_VALUES = [5, 10, 20]
