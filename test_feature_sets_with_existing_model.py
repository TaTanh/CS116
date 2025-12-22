"""Test model performance with different feature sets using existing trained model.

This script loads the best trained model (lightgbm_tuned) and evaluates it with:
1. 3 basic features (X1, X2, X3) - baseline from teacher
2. 5 features (X1-X5) - added recency & frequency  
3. 9 features (X1-X9) - added monetary & brand loyalty
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import List

import polars as pl

from src.recommender.config import (
    TRANSACTIONS_PATTERN,
    ITEMS_PATTERN,
    USERS_PATTERN,
    OUTPUT_DIR,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.recommender.data import (
    load_transactions,
    load_items,
    load_users,
)
from src.recommender.features import build_feature_label_table
from src.recommender.train import predict_and_rank
from src.recommender.metrics import evaluate_recommendations


def load_model(model_path: str):
    """Load a trained model from pickle file."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def test_feature_set_with_model(
    model,
    feature_label_table: pl.LazyFrame,
    feature_columns: List[str],
    feature_set_name: str,
    model_type: str = "lightgbm",
) -> dict:
    """Test model with specific feature set.
    
    Args:
        model: Pre-trained model
        feature_label_table: Full feature table
        feature_columns: List of features to use
        feature_set_name: Name for this feature set
        model_type: Type of model
        
    Returns:
        Dictionary with metrics and metadata
    """
    print(f"\n{'='*80}")
    print(f"Testing {feature_set_name}")
    print(f"Features ({len(feature_columns)}): {', '.join(feature_columns)}")
    print(f"{'='*80}\n")
    
    # Make predictions using only the specified features
    print("Making predictions...")
    predictions = predict_and_rank(
        model=model,
        feature_label_table=feature_label_table,
        feature_columns=feature_columns,
        top_k=10,
    )
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Get ground truth
    print("Extracting ground truth...")
    if isinstance(feature_label_table, pl.LazyFrame):
        ground_truth = feature_label_table.filter(pl.col("Y") == 1).select(["customer_id", "item_id"]).collect()
    else:
        ground_truth = feature_label_table.filter(pl.col("Y") == 1).select(["customer_id", "item_id"])
    
    print(f"Ground truth shape: {ground_truth.shape}")
    
    # Evaluate
    print("\nEvaluating recommendations...")
    metrics_dict = evaluate_recommendations(
        predictions=predictions,
        ground_truth=ground_truth,
        k_values=[10],
    )
    
    # Flatten metrics dict for easier access
    metrics = {}
    for metric_name, k_values in metrics_dict.items():
        for k, value in k_values.items():
            metrics[f"{metric_name}@{k}"] = value
    
    # Add metadata
    result = {
        "feature_set_name": feature_set_name,
        "num_features": len(feature_columns),
        "feature_columns": feature_columns,
        "model_type": model_type,
        **metrics
    }
    
    print(f"\nResults for {feature_set_name}:")
    print(f"  Precision@10: {metrics['precision@10']:.4f}")
    print(f"  Recall@10:    {metrics['recall@10']:.4f}")
    print(f"  NDCG@10:      {metrics['ndcg@10']:.4f}")
    print(f"  MAP@10:       {metrics['map@10']:.4f}")
    
    return result


def main():
    """Main execution function."""
    
    # Load the best trained model
    model_path = MODELS_DIR / "model_lightgbm_tuned_20251221_103746.pkl"
    print(f"Loading model from: {model_path}")
    model = load_model(str(model_path))
    print("Model loaded successfully!")
    
    print("\nLoading data...")
    transactions = load_transactions(TRANSACTIONS_PATTERN)
    items = load_items(ITEMS_PATTERN)
    users = load_users(USERS_PATTERN)
    
    # Define time windows (same as in training)
    print("\nBuilding features and labels...")
    begin_hist = datetime(2024, 1, 1)
    end_hist = datetime(2024, 11, 1)
    begin_recent = datetime(2024, 11, 1)
    end_recent = datetime(2024, 12, 1)
    
    print(f"Historical window: {begin_hist.date()} to {end_hist.date()}")
    print(f"Recent window:     {begin_recent.date()} to {end_recent.date()}")
    
    feature_label_table = build_feature_label_table(
        transactions=transactions,
        items=items,
        users=users,
        begin_hist=begin_hist,
        end_hist=end_hist,
        begin_recent=begin_recent,
        end_recent=end_recent,
    )
    
    # Define feature sets to test
    feature_sets = {
        "3_features_baseline": [
            "X1_brand_cnt_hist",
            "X2_age_group_cnt_hist", 
            "X3_category_cnt_hist",
        ],
        "5_features": [
            "X1_brand_cnt_hist",
            "X2_age_group_cnt_hist",
            "X3_category_cnt_hist",
            "X4_days_since_last_purchase",
            "X5_purchase_frequency",
        ],
        "9_features": [
            "X1_brand_cnt_hist",
            "X2_age_group_cnt_hist",
            "X3_category_cnt_hist",
            "X4_days_since_last_purchase",
            "X5_purchase_frequency",
            "X6_is_power_user",
            "X7_avg_items_per_purchase",
            "X8_top_brand_ratio",
            "X9_brand_diversity",
        ],
    }
    
    # Test each feature set
    all_results = []
    
    for set_name, features in feature_sets.items():
        result = test_feature_set_with_model(
            model=model,
            feature_label_table=feature_label_table,
            feature_columns=features,
            feature_set_name=set_name,
            model_type="lightgbm",
        )
        all_results.append(result)
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"feature_comparison_{timestamp}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("SUMMARY - Feature Set Comparison")
    print(f"{'='*80}\n")
    
    # Print comparison table
    print(f"{'Feature Set':<25} {'Features':<10} {'P@10':<10} {'R@10':<10} {'NDCG@10':<10} {'MAP@10':<10}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['feature_set_name']:<25} "
              f"{result['num_features']:<10} "
              f"{result['precision@10']:<10.4f} "
              f"{result['recall@10']:<10.4f} "
              f"{result['ndcg@10']:<10.4f} "
              f"{result['map@10']:<10.4f}")
    
    print(f"\nResults saved to: {output_path}")
    
    # Print analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")
    
    baseline = all_results[0]
    
    for i, result in enumerate(all_results[1:], 1):
        improvement_p10 = (result['precision@10'] - baseline['precision@10']) / baseline['precision@10'] * 100
        improvement_r10 = (result['recall@10'] - baseline['recall@10']) / baseline['recall@10'] * 100
        improvement_map = (result['map@10'] - baseline['map@10']) / baseline['map@10'] * 100
        
        print(f"{result['feature_set_name']} vs {baseline['feature_set_name']}:")
        print(f"  - Added {result['num_features'] - baseline['num_features']} features")
        print(f"  - Precision@10 change: {improvement_p10:+.2f}%")
        print(f"  - Recall@10 change:    {improvement_r10:+.2f}%")
        print(f"  - MAP@10 change:       {improvement_map:+.2f}%")
        print()


if __name__ == "__main__":
    main()
