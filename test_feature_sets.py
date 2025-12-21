"""Test model performance with different feature sets.

This script trains and evaluates models with:
1. 3 basic features (X1, X2, X3) - baseline from teacher
2. 5 features (X1-X5) - added recency & frequency
3. 9 features (X1-X9) - added monetary & brand loyalty
"""

import json
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
from src.recommender.features import build_features_and_labels
from src.recommender.train import train_model, predict_model
from src.recommender.metrics import evaluate_recommendations


def test_feature_set(
    feature_label_table: pl.LazyFrame,
    feature_columns: List[str],
    feature_set_name: str,
    model_type: str = "lightgbm",
) -> dict:
    """Test model with specific feature set.
    
    Args:
        feature_label_table: Full feature table
        feature_columns: List of features to use
        feature_set_name: Name for this feature set
        model_type: Type of model to train
        
    Returns:
        Dictionary with metrics and metadata
    """
    print(f"\n{'='*80}")
    print(f"Testing {feature_set_name}")
    print(f"Features ({len(feature_columns)}): {', '.join(feature_columns)}")
    print(f"{'='*80}\n")
    
    # Train model
    model = train_model(
        feature_label_table=feature_label_table,
        feature_columns=feature_columns,
        label_column="Y",
        model_type=model_type,
        random_state=RANDOM_STATE,
    )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"model_{model_type}_{feature_set_name}_{timestamp}.pkl"
    
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predict_model(
        model=model,
        feature_table=feature_label_table,
        feature_columns=feature_columns,
        model_type=model_type,
    )
    
    # Evaluate
    print("\nEvaluating recommendations...")
    metrics = evaluate_recommendations(
        predictions=predictions,
        k_values=[5, 10, 20],
    )
    
    # Add metadata
    result = {
        "feature_set_name": feature_set_name,
        "num_features": len(feature_columns),
        "feature_columns": feature_columns,
        "model_type": model_type,
        "model_path": str(model_path),
        "timestamp": timestamp,
        **metrics
    }
    
    print(f"\nResults for {feature_set_name}:")
    print(f"  Precision@5:  {metrics['precision@5']:.4f}")
    print(f"  Precision@10: {metrics['precision@10']:.4f}")
    print(f"  Precision@20: {metrics['precision@20']:.4f}")
    print(f"  Recall@5:     {metrics['recall@5']:.4f}")
    print(f"  Recall@10:    {metrics['recall@10']:.4f}")
    print(f"  Recall@20:    {metrics['recall@20']:.4f}")
    print(f"  MAP@20:       {metrics['map@20']:.4f}")
    
    return result


def main():
    """Main execution function."""
    print("Loading data...")
    transactions = load_transactions(TRANSACTIONS_PATTERN)
    items = load_items(ITEMS_PATTERN)
    users = load_users(USERS_PATTERN)
    
    # Define time windows (same as in train_all_models.py)
    print("\nBuilding features and labels...")
    begin_hist = datetime(2018, 1, 1)
    end_hist = datetime(2018, 9, 1)
    begin_recent = datetime(2018, 9, 1)
    end_recent = datetime(2018, 10, 1)
    
    feature_label_table = build_features_and_labels(
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
        result = test_feature_set(
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
    print(f"{'Feature Set':<25} {'Features':<10} {'P@5':<8} {'P@10':<8} {'P@20':<8} {'R@20':<8} {'MAP@20':<8}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['feature_set_name']:<25} "
              f"{result['num_features']:<10} "
              f"{result['precision@5']:<8.4f} "
              f"{result['precision@10']:<8.4f} "
              f"{result['precision@20']:<8.4f} "
              f"{result['recall@20']:<8.4f} "
              f"{result['map@20']:<8.4f}")
    
    print(f"\nResults saved to: {output_path}")
    
    # Print analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")
    
    baseline = all_results[0]
    
    for i, result in enumerate(all_results[1:], 1):
        improvement_p5 = (result['precision@5'] - baseline['precision@5']) / baseline['precision@5'] * 100
        improvement_r20 = (result['recall@20'] - baseline['recall@20']) / baseline['recall@20'] * 100
        improvement_map = (result['map@20'] - baseline['map@20']) / baseline['map@20'] * 100
        
        print(f"{result['feature_set_name']} vs {baseline['feature_set_name']}:")
        print(f"  - Added {result['num_features'] - baseline['num_features']} features")
        print(f"  - Precision@5 change:  {improvement_p5:+.2f}%")
        print(f"  - Recall@20 change:    {improvement_r20:+.2f}%")
        print(f"  - MAP@20 change:       {improvement_map:+.2f}%")
        print()


if __name__ == "__main__":
    main()
