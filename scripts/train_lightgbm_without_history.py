"""
Train LightGBM WITHOUT historical features (X1-X3)
Only use recent/behavioral features: X4-X13
Compare performance with model that has history
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
print("TRAINING LIGHTGBM - WITHOUT HISTORY (X4-X13 ONLY)")
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
    print("Run train_lightgbm_parameter.py first to build features.")
    exit(1)

# Feature columns - WITHOUT X1, X2, X3 (historical features)
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

print(f"\n[INFO] Using {len(feature_cols)} features (WITHOUT history):")
print(f"  Excluded: X1, X2, X3 (historical features)")
print(f"  Included: X4-X13 (recent/behavioral features)")

# Ground truth
ground_truth = features.filter(pl.col('Y') == 1).select(['customer_id', 'item_id'])
print(f"\nGround truth: {ground_truth.shape[0]:,} positive pairs")

# Create output directories
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

# Train LightGBM with TUNED parameters (same as with-history model)
print("\n" + "="*70)
print("TRAINING LIGHTGBM - WITHOUT HISTORY")
print("="*70)

# Tuned hyperparameters (same as with-history for fair comparison)
tuned_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": 8,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 100,
    "min_child_weight": 0.001,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42,
}

print("\n[1/4] Training LightGBM WITHOUT history features...")
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

print("\nMetrics (WITHOUT history):")
for metric_name, values in metrics.items():
    print(f"  {metric_name.upper()}:")
    for k, score in values.items():
        print(f"    @{k:2d}: {score:.4f}")

# Save outputs
print("\n[4/4] Saving outputs...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = f"outputs/models/model_lightgbm_without_history_{timestamp}.pkl"
predictions_path = f"outputs/predictions/predictions_lightgbm_without_history_{timestamp}.parquet"
metrics_path = f"outputs/metrics_lightgbm_without_history_{timestamp}.json"

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

# Compare with WITH-history model
print("\n[COMPARISON]")
print("WITH history (X1-X13) - from previous run:")
print("  Precision@10: 0.0415")
print("  NDCG@10: 0.1195")
print(f"\nWITHOUT history (X4-X13) - this run:")
print(f"  Precision@10: {metrics['precision'][10]:.4f}")
print(f"  NDCG@10: {metrics['ndcg'][10]:.4f}")

# Calculate impact
impact_p10 = (metrics['precision'][10] - 0.0415) / 0.0415 * 100
impact_ndcg = (metrics['ndcg'][10] - 0.1195) / 0.1195 * 100
print(f"\nImpact of removing history:")
print(f"  Precision@10: {impact_p10:+.2f}%")
print(f"  NDCG@10: {impact_ndcg:+.2f}%")

if impact_p10 < -5:
    print(f"\nHistorical features (X1-X3) are IMPORTANT!")
    print(f"  Removing them causes significant performance drop")
elif impact_p10 > 5:
    print(f"\nHistorical features are NOT important!")
    print(f"  Recent features (X4-X13) are sufficient")
else:
    print(f"\nHistorical features have MINOR impact")
    print(f"  Both approaches work similarly")
