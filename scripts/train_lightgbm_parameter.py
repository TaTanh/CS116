"""
Train LightGBM with TUNED hyperparameters (60% customers)
Optimized parameters for better performance
"""

from datetime import datetime
import polars as pl
from src.recommender import (
    load_transactions, load_items, load_users,
    build_feature_label_table,
    train_model, predict_and_rank, save_model, evaluate_ranking
)
import json
import os

print("="*70)
print("TRAINING LIGHTGBM - TUNED PARAMETERS (60% CUSTOMERS)")
print("="*70)

# Use cached features from previous run
features_cache = "outputs/temp/features_cache_full.parquet"
if os.path.exists(features_cache):
    print(f"\nUsing cached features from {features_cache}")
    print("  Loading features...")
    features = pl.read_parquet(features_cache)
    print(f"  Features loaded: {features.shape}")
else:
    print("\nERROR: No cached features found!")
    print("Run train_lightgbm_full.py first to build features.")
    exit(1)

# Feature columns
feature_cols = [
    'X1_brand_cnt_hist', 'X2_age_group_cnt_hist', 'X3_category_cnt_hist',
    'X4_days_since_last_purchase', 'X5_purchase_frequency', 'X6_is_power_user',
    'X7_avg_items_per_purchase', 'X8_top_brand_ratio', 'X9_brand_diversity',
    'X10_category_diversity_score', 'X11_purchase_day_mode', 'X12_is_new_customer',
    'X13_avg_item_popularity'
]

# Ground truth
ground_truth = features.filter(pl.col('Y') == 1).select(['customer_id', 'item_id'])
print(f"\nGround truth: {ground_truth.shape[0]:,} positive pairs")

# Create output directories
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

# Train LightGBM with TUNED parameters
print("\n" + "="*70)
print("TRAINING LIGHTGBM - TUNED HYPERPARAMETERS")
print("="*70)

# Tuned hyperparameters (better than default)
tuned_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,              # Increased from 31 (more complex trees)
    "max_depth": 8,                # Increased from -1 (control overfitting)
    "learning_rate": 0.03,         # Decreased from 0.05 (more careful learning)
    "feature_fraction": 0.8,       # Same
    "bagging_fraction": 0.7,       # Decreased from 0.8 (more regularization)
    "bagging_freq": 5,             # Same
    "min_child_samples": 100,      # Added (prevent overfitting)
    "min_child_weight": 0.001,     # Added
    "reg_alpha": 0.1,              # Added L1 regularization
    "reg_lambda": 0.1,             # Added L2 regularization
    "verbose": -1,
    "seed": 42,
}

print("\nTuned hyperparameters:")
for key, value in tuned_params.items():
    if key not in ['objective', 'metric', 'boosting_type', 'verbose', 'seed']:
        print(f"  {key}: {value}")

print("\n[1/4] Training LightGBM with tuned params...")
model = train_model(
    features, 
    feature_cols, 
    label_column='Y',
    model_type='lightgbm',
    model_params=tuned_params,
    random_state=42
)
print("LightGBM trained successfully")

# Generate predictions
print("\n[2/4] Generating predictions...")
predictions = predict_and_rank(
    model=model,
    feature_label_table=features,
    feature_columns=feature_cols,
    top_k=20
)
print(f"Predictions: {predictions.shape}")

# Evaluate
print("\n[3/4] Evaluating...")
metrics = evaluate_ranking(
    predictions=predictions,
    ground_truth=ground_truth,
    k_values=[5, 10, 20]
)

print("\nMetrics (with tuned params):")
for metric_name, values in metrics.items():
    print(f"  {metric_name.upper()}:")
    for k, score in values.items():
        print(f"    @{k:2d}: {score:.4f}")

# Save outputs
print("\n[4/4] Saving outputs...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = f"outputs/models/model_lightgbm_tuned_{timestamp}.pkl"
predictions_path = f"outputs/predictions/predictions_lightgbm_tuned_{timestamp}.parquet"
metrics_path = f"outputs/metrics_lightgbm_tuned_{timestamp}.json"

save_model(model, model_path)
predictions.write_parquet(predictions_path)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Model saved: {model_path}")
print(f"Predictions saved: {predictions_path}")
print(f"Metrics saved: {metrics_path}")

print("\n" + "="*70)
print("DONE!")
print("="*70)

# Compare with default params
print("\n[COMPARISON]")
print("Default params (from previous run):")
print("  Precision@10: 0.0412")
print("  NDCG@10: 0.1190")
print(f"\nTuned params (this run):")
print(f"  Precision@10: {metrics['precision'][10]:.4f}")
print(f"  NDCG@10: {metrics['ndcg'][10]:.4f}")

improvement_p10 = (metrics['precision'][10] - 0.0412) / 0.0412 * 100
improvement_ndcg = (metrics['ndcg'][10] - 0.1190) / 0.1190 * 100
print(f"\nImprovement:")
print(f"  Precision@10: {improvement_p10:+.1f}%")
print(f"  NDCG@10: {improvement_ndcg:+.1f}%")

print(f"\nNext step:")
print(f"  Update optimize_submission.py predictions_file to:")
print(f"  {predictions_path}")
