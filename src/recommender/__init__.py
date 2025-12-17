"""Time-based purchase prediction recommender system."""

__version__ = "0.1.0"

from .config import DATA_DIR, TRANSACTIONS_PATH, ITEMS_PATH, USERS_PATH
from .data import load_transactions, load_items, load_users
from .utils import list_parquet_files, explore_dataset, load_any_parquet
from .features import (
    build_feature_label_table,
    add_baby_age_feature,
    build_item_segment_step1,
    build_customer_segment_from_step1,
)
from .candidates import (
    generate_candidates,
    build_item_cooccurrence,
    generate_candidates_from_cooc,
)
from .metrics import precision_at_k, ndcg_at_k
from .train import train_model, predict_and_rank, evaluate_ranking, save_model, load_model, get_feature_importance

__all__ = [
    # Config
    "DATA_DIR",
    "TRANSACTIONS_PATH",
    "ITEMS_PATH",
    "USERS_PATH",
    # Data loading
    "load_transactions",
    "load_items",
    "load_users",
    "list_parquet_files",
    "explore_dataset",
    "load_any_parquet",
    # Features
    "build_feature_label_table",
    "add_baby_age_feature",
    "build_item_segment_step1",
    "build_customer_segment_from_step1",
    # Candidates
    "generate_candidates",
    "build_item_cooccurrence",
    "generate_candidates_from_cooc",
    # Metrics
    "precision_at_k",
    "ndcg_at_k",
    # Training
    "train_model",
    "predict_and_rank",
    "evaluate_ranking",
    "save_model",
    "load_model",
    "get_feature_importance",
]
