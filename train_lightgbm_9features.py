"""
Train LightGBM with 9 features (X1-X9)
- Baseline + Recency & Frequency + Monetary & Brand Loyalty
- Sample 40% data to avoid memory issues
"""

from datetime import datetime
import polars as pl
from src.recommender import (
    load_transactions, load_items, load_users,
    build_feature_label_table,
    train_model, predict_and_rank
)
from src.recommender.metrics import evaluate_recommendations
import json
import os
import pickle

print("="*70)
print("TRAIN LIGHTGBM - 9 FEATURES")
print("="*70)

# Load data
print("\n[1] Loading data...")
transactions = load_transactions()
items = load_items()
users = load_users()
print("Data loaded")

# Define time windows
print("\n[2] Defining time windows...")
begin_hist = datetime(2024, 1, 1)
end_hist = datetime(2024, 11, 1)
begin_recent = datetime(2024, 11, 1)
end_recent = datetime(2024, 12, 1)
print(f"Hist: {begin_hist.date()} to {end_hist.date()}")
print(f"Recent: {begin_recent.date()} to {end_recent.date()}")

# Sample 40% customers to avoid memory issues
print("\n[3] Sampling 40% of customers...")
all_customers = transactions.select("customer_id").unique().collect()
import random
random.seed(42)
n_sample = int(len(all_customers) * 0.4)
sampled_customers = all_customers.sample(n=n_sample)
print(f"Sampled {len(sampled_customers):,} customers")

transactions_sampled = transactions.join(
    sampled_customers.lazy(),
    on="customer_id",
    how="inner"
)

# Build features
print("\n[4] Building features...")
features_lazy = build_feature_label_table(
    transactions_sampled, items, users,
    begin_hist, end_hist,
    begin_recent, end_recent
)

print("\n[5] Collecting features with STREAMING...")
features = features_lazy.collect(streaming=True)
print(f"Features: {features.shape}")

# Feature columns - 9 features (baseline + recency/frequency + monetary/brand)
feature_cols = [
    'X1_brand_cnt_hist',
    'X2_age_group_cnt_hist',
    'X3_category_cnt_hist',
    'X4_days_since_last_purchase',
    'X5_purchase_frequency',
    'X6_is_power_user',
    'X7_avg_items_per_purchase',
    'X8_top_brand_ratio',
    'X9_brand_diversity',
]

print(f"\n[6] Using {len(feature_cols)} features: {', '.join(feature_cols)}")

# Train model
print("\n[7] Training LightGBM...")
model = train_model(
    features, 
    feature_cols, 
    label_column='Y',
    model_type='lightgbm',
    random_state=42
)
print("Model trained successfully")

# Generate predictions
print("\n[8] Generating predictions...")
predictions = predict_and_rank(
    model=model,
    feature_label_table=features,
    feature_columns=feature_cols,
    top_k=20
)
print(f"Predictions: {predictions.shape}")

# Ground truth
ground_truth = features.filter(pl.col('Y') == 1).select(['customer_id', 'item_id'])
print(f"Ground truth: {ground_truth.shape[0]:,} positive pairs")

# Evaluate
print("\n[9] Evaluating...")
metrics = evaluate_recommendations(
    predictions=predictions,
    ground_truth=ground_truth,
    k_values=[5, 10, 20]
)

print(f"\nMetrics for 9 features:")
for metric_name, values in metrics.items():
    print(f"  {metric_name.upper()}:")
    for k, score in values.items():
        print(f"    @{k}: {score:.4f}")

# Save outputs
print("\n[10] Saving outputs...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

model_path = f"outputs/models/model_lightgbm_9features_{timestamp}.pkl"
predictions_path = f"outputs/predictions/predictions_lightgbm_9features_{timestamp}.parquet"
metrics_path = f"outputs/metrics_lightgbm_9features_{timestamp}.json"

with open(model_path, "wb") as f:
    pickle.dump(model, f)
predictions.write_parquet(predictions_path)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Model saved: {model_path}")
print(f"Predictions saved: {predictions_path}")
print(f"Metrics saved: {metrics_path}")

print("\n" + "="*70)
print("DONE - 9 FEATURES")
print("="*70)
