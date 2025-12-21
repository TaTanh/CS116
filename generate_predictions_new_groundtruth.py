"""
Generate predictions using best existing model for new groundtruth
NO TRAINING - just use existing LightGBM tuned model
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
print("GENERATE PREDICTIONS - NEW GROUNDTRUTH (NO TRAINING)")
print("="*70)

# 1. Load best model
model_path = "outputs/models/model_lightgbm_tuned_20251221_103746.pkl"
print(f"\n[1] Loading best model...")
print(f"Model: {model_path}")

with open(model_path, "rb") as f:
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

# Use same time windows as original model (train on 2024 data)
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

# 4. Generate predictions
print("\n[4] Generating predictions...")
feature_cols = [
    'X1_brand_cnt_hist', 'X2_age_group_cnt_hist', 'X3_category_cnt_hist',
    'X4_days_since_last_purchase', 'X5_purchase_frequency', 'X6_is_power_user',
    'X7_avg_items_per_purchase', 'X8_top_brand_ratio', 'X9_brand_diversity',
    'X10_category_diversity_score', 'X11_purchase_day_mode', 'X12_is_new_customer',
    'X13_avg_item_popularity'
]

predictions = predict_and_rank(
    model=model,
    feature_label_table=features,
    feature_columns=feature_cols,
    top_k=10
)

print(f"Predictions shape: {predictions.shape}")
print(f"Customers with predictions: {predictions['customer_id'].n_unique():,}")

# 5. Save predictions
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pred_path = f"outputs/predictions/predictions_new_groundtruth_{timestamp}.parquet"
predictions.write_parquet(pred_path)
print(f"\n[5] Predictions saved to: {pred_path}")

# 6. Quick evaluation
print("\n[6] Quick evaluation on new groundtruth...")
import numpy as np

metrics = {}
for k in [5, 10, 20]:
    # Get top-k predictions
    top_k_predictions = predictions.filter(pl.col("rank") <= k)
    
    # Calculate metrics - these functions handle dict groundtruth
    precision = 0.0
    recall = 0.0
    ndcg = 0.0
    
    # Calculate per customer and average
    num_customers = 0
    total_precision = 0.0
    total_recall = 0.0
    total_ndcg = 0.0
    
    for customer_id in groundtruth.keys():
        # Get predictions for this customer
        cust_preds = top_k_predictions.filter(pl.col("customer_id") == customer_id)
        if cust_preds.shape[0] == 0:
            continue
            
        predicted_items = cust_preds["item_id"].to_list()
        actual_items = groundtruth[customer_id]
        
        # Skip if actual_items is a string (format issue)
        if isinstance(actual_items, str):
            continue
        # Handle nested list
        if len(actual_items) > 0 and isinstance(actual_items[0], list):
            actual_items = actual_items[0]
        # Convert to set for faster lookup
        actual_set = set(str(item) for item in actual_items)
        predicted_set = set(str(item) for item in predicted_items)
        
        # Precision
        if len(predicted_set) > 0:
            hits = len(predicted_set & actual_set)
            total_precision += hits / len(predicted_set)
            
            # Recall
            if len(actual_set) > 0:
                total_recall += hits / len(actual_set)
                
            # NDCG (simplified)
            dcg = sum([1.0 / np.log2(i + 2) if pred in actual_set else 0 
                      for i, pred in enumerate(predicted_items)])
            idcg = sum([1.0 / np.log2(i + 2) 
                       for i in range(min(len(actual_set), k))])
            if idcg > 0:
                total_ndcg += dcg / idcg
                
        num_customers += 1
    
    # Average metrics
    if num_customers > 0:
        precision = total_precision / num_customers
        recall = total_recall / num_customers
        ndcg = total_ndcg / num_customers
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics[f'precision@{k}'] = precision
    metrics[f'recall@{k}'] = recall
    metrics[f'ndcg@{k}'] = ndcg
    metrics[f'f1@{k}'] = f1

print("\nMetrics on NEW groundtruth:")
for k in [5, 10, 20]:
    print(f"  K={k}:")
    print(f"    Precision: {metrics[f'precision@{k}']:.4f}")
    print(f"    Recall:    {metrics[f'recall@{k}']:.4f}")
    print(f"    NDCG:      {metrics[f'ndcg@{k}']:.4f}")
    print(f"    F1:        {metrics[f'f1@{k}']:.4f}")

# Save metrics
metrics_path = f"outputs/metrics_new_groundtruth_{timestamp}.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to: {metrics_path}")

print("\n" + "="*70)
print("COMPLETED!")
print("="*70)
print(f"Predictions file: {pred_path}")
print(f"Next step: Run optimize_submission.py to create submission")
