"""
Generate predictions using WITHOUT HISTORY model for new groundtruth
NO TRAINING - just use existing model WITHOUT history (X4-X13)
"""

from datetime import datetime
import polars as pl
import pickle
import os
import json
from src.recommender import (
    load_transactions, load_items, load_users,
    build_feature_label_table,
    predict_and_rank
)

print("="*70)
print("GENERATE PREDICTIONS - WITHOUT HISTORY (X4-X13)")
print("="*70)

# 1. Load WITHOUT HISTORY model
model_file = "outputs/models/model_lightgbm_without_history_20251222_102730.pkl"
print(f"\n[1] Loading WITHOUT HISTORY model...")
print(f"Model: {model_file}")

if not os.path.exists(model_file):
    print(f"ERROR: Model not found!")
    print("Please run train_lightgbm_without_history.py first")
    exit(1)

with open(model_file, "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully")

# 2. Load groundtruth to get customers to predict
print("\n[2] Loading new groundtruth...")
with open("groundtruth.pkl", "rb") as f:
    groundtruth = pickle.load(f)

groundtruth_customers = set(groundtruth.keys())
print(f"Groundtruth customers: {len(groundtruth_customers):,}")

# 3. Build features for new groundtruth customers
print("\n[3] Building features for new groundtruth customers...")
print("Loading data schemas...")
transactions = load_transactions()
items = load_items()
users = load_users()

# Use same time windows as WITH HISTORY model
begin_hist = datetime(2024, 1, 1)
end_hist = datetime(2024, 11, 1)
begin_recent = datetime(2024, 11, 1)
end_recent = datetime(2024, 12, 1)

print(f"Hist: {begin_hist.date()} to {end_hist.date()}")
print(f"Recent: {begin_recent.date()} to {end_recent.date()}")

# Filter to only groundtruth customers
print(f"\nFiltering to {len(groundtruth_customers):,} groundtruth customers...")
transactions = transactions.filter(pl.col("customer_id").is_in(list(groundtruth_customers)))

print("\nBuilding features...")
features_lazy = build_feature_label_table(
    transactions, items, users,
    begin_hist, end_hist,
    begin_recent, end_recent
)

print("Collecting features (this may take a few minutes)...")
features = features_lazy.collect(streaming=True)
print(f"Features shape: {features.shape}")

# 4. Generate predictions - WITHOUT HISTORY (X4-X13 only)
print("\n[4] Generating predictions WITHOUT history...")
feature_cols = [
    # 'X1_brand_cnt_hist',           # EXCLUDED
    # 'X2_age_group_cnt_hist',       # EXCLUDED
    # 'X3_category_cnt_hist',        # EXCLUDED
    'X4_days_since_last_purchase',
    'X5_purchase_frequency',
    'X6_is_power_user',
    'X7_avg_items_per_purchase',
    'X8_top_brand_ratio',
    'X9_brand_diversity',
    'X10_category_diversity_score',
    'X11_purchase_day_mode',
    'X12_is_new_customer',
    'X13_avg_item_popularity'
]

print(f"Using {len(feature_cols)} features (X4-X13, NO history)")

predictions = predict_and_rank(
    model=model,
    feature_label_table=features,
    feature_columns=feature_cols,
    top_k=10
)

print(f"Predictions shape: {predictions.shape}")
print(f"Customers with predictions: {predictions['customer_id'].n_unique():,}")

# 5. Save predictions
pred_path = "outputs/predictions/predictions_lightgbm_without_history_newgroundtruth.parquet"
predictions.write_parquet(pred_path)
print(f"\n[5] Predictions saved to: {pred_path}")

print("\n" + "="*70)
print("COMPLETED!")
print("="*70)
print(f"Predictions file: {pred_path}")
