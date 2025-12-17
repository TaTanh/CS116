"""
Quick training script for recommender system.
This works with LazyFrame to avoid loading all 36M transactions into memory.
"""

from datetime import datetime
import polars as pl
from src.recommender import (
    load_transactions, load_items, load_users,
    build_feature_label_table,
    train_model, predict_and_rank
)

print("="*70)
print("RECOMMENDER SYSTEM - TRAINING PIPELINE")
print("="*70)

# 1. Load data (LazyFrame - not loaded into memory yet)
print("\n[1/6] Loading data schemas...")
transactions = load_transactions()
items = load_items()
users = load_users()
print("✓ Data schemas loaded (LazyFrame)")

# 2. Define time windows (adjust these based on your data)
print("\n[2/6] Defining time windows...")
begin_hist = datetime(2024, 1, 1)      # Historical period start
end_hist = datetime(2024, 10, 1)       # Historical period end
begin_recent = datetime(2024, 10, 1)   # Recent period start (for labels)
end_recent = datetime(2024, 11, 1)     # Recent period end
print(f"✓ Hist period: {begin_hist.date()} to {end_hist.date()}")
print(f"✓ Recent period: {begin_recent.date()} to {end_recent.date()}")

# 3. Build features (still LazyFrame - efficient!)
print("\n[3/6] Building features...")
print("   Note: This creates a query plan, no data loaded yet...")
print("   ⚠️  Using 20% customer sample to prevent memory crash...")

# Sample customers to reduce data size (from 244K to ~49K customers)
# This reduces candidate pairs from potentially 100M+ to ~20M
sampled_customers = (
    transactions
    .select("customer_id")
    .unique()
    .filter((pl.col("customer_id").hash(seed=42) % 5) == 0)  # 20% sample
)

# Filter transactions to sampled customers only
transactions_sampled = transactions.join(sampled_customers, on="customer_id", how="inner")

features_lazy = build_feature_label_table(
    transactions_sampled, items, users,
    begin_hist, end_hist,
    begin_recent, end_recent
)
print(f"✓ Feature query plan created (with 20% customer sample)")

# 4. Collect features with STREAMING (prevents memory crash)
print("\n[4/6] Collecting features with STREAMING (prevents memory crash)...")
print("   ⚠️  Processing 36M+ transactions in chunks")
print("   This may take 10-15 minutes but uses <3GB RAM...")
print("   Progress: ", end="", flush=True)

# STREAM PROCESSING: Use collect(streaming=True) for chunked processing
try:
    features = features_lazy.collect(streaming=True)
    print("✓ Collected with streaming mode")
except Exception as e:
    print(f"\n   Warning: Streaming mode failed ({str(e)})")
    print("   Falling back to standard collect (may use more RAM)...")
    features = features_lazy.collect()
    print("✓ Collected successfully")

print(f"✓ Features collected: {features.shape}")
pos_count = features.filter(pl.col('Y')==1).shape[0]
neg_count = features.filter(pl.col('Y')==0).shape[0]
print(f"   Positive samples (Y=1): {pos_count:,}")
print(f"   Negative samples (Y=0): {neg_count:,}")
print(f"   Class balance: {pos_count/(pos_count+neg_count):.2%} positive")

# 5. Train model
print("\n[5/6] Training model...")
feature_cols = [
    'X1_brand_cnt_hist', 'X2_age_group_cnt_hist', 'X3_category_cnt_hist',
    'X4_days_since_last_purchase', 'X5_purchase_frequency', 'X6_is_power_user',
    'X7_avg_items_per_purchase', 'X8_top_brand_ratio', 'X9_brand_diversity',
    'X10_category_diversity_score', 'X11_purchase_day_mode', 'X12_is_new_customer',
    'X13_avg_item_popularity'
]
print(f"   Using {len(feature_cols)} features from EDA insights")
model = train_model(
    features, 
    feature_cols, 
    label_column='Y',
    model_type='logistic',  # Change to 'lightgbm' if you want
    random_state=42
)
print("✓ Model trained successfully")

# 6. Generate predictions and evaluate
print("\n[6/6] Generating predictions (Top 20 per customer)...")
predictions = predict_and_rank(
    model=model,
    feature_label_table=features,
    feature_columns=feature_cols,
    top_k=20
)
print(f"✓ Predictions generated: {predictions.shape}")
print(f"\nSample predictions:")
print(predictions.head(10))

# Evaluate
print("\n" + "="*70)
print("EVALUATION")
print("="*70)
ground_truth = features.filter(pl.col('Y') == 1).select(['customer_id', 'item_id'])
print(f"Ground truth size: {ground_truth.shape[0]:,} positive pairs")

# Simple precision calculation
from src.recommender import evaluate_ranking
metrics = evaluate_ranking(
    predictions=predictions,
    ground_truth=ground_truth,
    k_values=[5, 10, 20]
)

print("\nMetrics:")
for metric_name, values in metrics.items():
    print(f"\n{metric_name.upper()}:")
    for k, score in values.items():
        print(f"  @{k:2d}: {score:.4f}")

# Save outputs
print("\n" + "="*70)
print("SAVING OUTPUTS")
print("="*70)

# Save model
print("\n[7/7] Saving model and predictions...")
from src.recommender import save_model
import json
import os

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)
model_path = f"outputs/models/model_{timestamp}.pkl"
predictions_path = f"outputs/predictions/predictions_{timestamp}.parquet"
metrics_path = f"outputs/metrics_{timestamp}.json"

save_model(model, model_path)
print(f"✓ Model saved to: {model_path}")

predictions.write_parquet(predictions_path)
print(f"✓ Predictions saved to: {predictions_path}")

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Metrics saved to: {metrics_path}")

print("\n" + "="*70)
print("DONE! Model training completed successfully.")
print("="*70)
print("\nSaved files:")
print(f"  - Model: {model_path}")
print(f"  - Predictions: {predictions_path}")
print(f"  - Metrics: {metrics_path}")
print("\nNext steps:")
print("  1. Adjust time windows if needed")
print("  2. Try model_type='lightgbm' for better performance")
print("  3. Add more features in features.py")
print("  4. Tune model hyperparameters")
