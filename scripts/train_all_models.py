"""
Train all remaining models: LightGBM, XGBoost, Random Forest
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

# Models to train (all 4 models)
models_to_train = ['logistic', 'lightgbm', 'xgboost', 'random_forest']

print("="*70)
print("TRAINING ALL 4 MODELS")
print("="*70)
print(f"Models: {', '.join(models_to_train)}")

# Check if features already exist
features_cache = "outputs/temp/features_cache.parquet"
if os.path.exists(features_cache):
    print(f"\nUsing cached features from {features_cache}")
    print("  Loading features...")
    features = pl.read_parquet(features_cache)
    print(f"  Features loaded: {features.shape}")
else:
    print("\nNo cached features found. Building features first...")
    print("  This will take 10-15 minutes...")
    
    # Load data
    print("\n[1] Loading data schemas...")
    transactions = load_transactions()
    items = load_items()
    users = load_users()
    print("Data schemas loaded")
    
    # Define time windows (Option 3: Use maximum data)
    print("\n[2] Defining time windows (Option 3)...")
    begin_hist = datetime(2024, 1, 1)
    end_hist = datetime(2024, 11, 1)      # Extended to Nov (more data)
    begin_recent = datetime(2024, 11, 1)
    end_recent = datetime(2024, 12, 1)    # Predict Dec purchases
    print(f"Hist: {begin_hist.date()} to {end_hist.date()}")
    print(f"Recent: {begin_recent.date()} to {end_recent.date()}")
    
    # Use ALL customers (100% - no sampling)
    print("\n[3] Using ALL customers (no sampling)...")
    
    # Build features
    print("\n[4] Building features...")
    features_lazy = build_feature_label_table(
        transactions, items, users,
        begin_hist, end_hist,
        begin_recent, end_recent
    )
    
    print("\n[5] Collecting features with STREAMING...")
    features = features_lazy.collect(streaming=True)
    print(f"Features: {features.shape}")
    
    # Cache features
    os.makedirs("outputs/temp", exist_ok=True)
    features.write_parquet(features_cache)
    print(f"Features cached to {features_cache}")

# Feature columns
feature_cols = [
    'X1_brand_cnt_hist', 'X2_age_group_cnt_hist', 'X3_category_cnt_hist',
    'X4_days_since_last_purchase', 'X5_purchase_frequency', 'X6_is_power_user',
    'X7_avg_items_per_purchase', 'X8_top_brand_ratio', 'X9_brand_diversity',
    'X10_category_diversity_score', 'X11_purchase_day_mode', 'X12_is_new_customer',
    'X13_avg_item_popularity'
]

# Ground truth for evaluation
ground_truth = features.filter(pl.col('Y') == 1).select(['customer_id', 'item_id'])
print(f"\nGround truth: {ground_truth.shape[0]:,} positive pairs")

# Create output directories
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

# Train each model
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

all_results = {}

for model_type in models_to_train:
    print(f"\n{'='*70}")
    print(f"MODEL: {model_type.upper()}")
    print(f"{'='*70}")
    
    try:
        # Train model
        print(f"\n[1/{len(models_to_train)}] Training {model_type}...")
        model = train_model(
            features, 
            feature_cols, 
            label_column='Y',
            model_type=model_type,
            random_state=42
        )
        print(f"{model_type} trained successfully")
        
        # Generate predictions
        print(f"\n[2/{len(models_to_train)}] Generating predictions...")
        predictions = predict_and_rank(
            model=model,
            feature_label_table=features,
            feature_columns=feature_cols,
            top_k=20
        )
        print(f"Predictions: {predictions.shape}")
        
        # Evaluate
        print(f"\n[3/{len(models_to_train)}] Evaluating...")
        metrics = evaluate_ranking(
            predictions=predictions,
            ground_truth=ground_truth,
            k_values=[5, 10, 20]
        )
        
        print(f"\nMetrics for {model_type}:")
        for metric_name, values in metrics.items():
            print(f"  {metric_name.upper()}:")
            for k, score in values.items():
                print(f"    @{k:2d}: {score:.4f}")
        
        all_results[model_type] = metrics
        
        # Save outputs
        print(f"\n[4/{len(models_to_train)}] Saving outputs...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = f"outputs/models/model_{model_type}_{timestamp}.pkl"
        predictions_path = f"outputs/predictions/predictions_{model_type}_{timestamp}.parquet"
        metrics_path = f"outputs/metrics_{model_type}_{timestamp}.json"
        
        save_model(model, model_path)
        predictions.write_parquet(predictions_path)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Model saved: {model_path}")
        print(f"Predictions saved: {predictions_path}")
        print(f"Metrics saved: {metrics_path}")
        
    except Exception as e:
        print(f"\nERROR training {model_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        all_results[model_type] = {"error": str(e)}

# Summary
print("\n" + "="*70)
print("SUMMARY - ALL MODELS")
print("="*70)

for model_type, result in all_results.items():
    print(f"\n{model_type.upper()}:")
    if "error" in result:
        print(f"  FAILED: {result['error']}")
    else:
        for metric_name, values in result.items():
            print(f"  {metric_name}@10: {values.get(10, 'N/A'):.4f}")

print("\n" + "="*70)
print("ALL DONE!")
print("="*70)