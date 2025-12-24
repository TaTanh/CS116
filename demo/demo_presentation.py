"""
DEMO PRESENTATION - Product Recommendation System
Run: python demo_presentation.py

Quick demo showing:
1. Customer profile
2. Top 10 recommendations
3. Evaluation results
"""

import pickle
import polars as pl
import os
import random
import numpy as np

def format_number(n):
    """Format number with thousand separators"""
    return f"{n:,}"

print("="*70)
print("PRODUCT RECOMMENDATION SYSTEM - LIVE DEMO")
print("="*70)

# Load model and predictions
print("\n[1/4] Loading model and predictions...")
model_path = "../outputs/models/model_lightgbm_tuned_20251221_103746.pkl"
predictions_path = "../outputs/predictions/predictions_lightgbm_tuned_20251221_103746.parquet"

if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    exit(1)

model = pickle.load(open(model_path, "rb"))
predictions = pl.read_parquet(predictions_path)
print("Model loaded successfully")
print(f"Predictions loaded: {format_number(predictions.shape[0])} rows")

# Load groundtruth
print("\n[2/4] Loading groundtruth...")
with open("../groundtruth.pkl", "rb") as f:
    groundtruth = pickle.load(f)
print(f"Groundtruth loaded: {format_number(len(groundtruth))} customers")

# Find BEST, WORST, and RANDOM cases
print("\n[3/4] Finding best, worst, and random demo cases...")

# Sample 1000 customers to analyze
sample_customers = (
    predictions
    .select("customer_id")
    .unique()
    .filter(pl.col("customer_id").is_in(list(groundtruth.keys())))
    .head(1000)
    .to_series()
    .to_list()
)

best_precision = 0
worst_precision = 1.0
best_customer = None
worst_customer = None
best_top10 = None
worst_top10 = None
best_hits = 0
worst_hits = 0
best_actual = []
worst_actual = []

for cust_id in sample_customers:
    # Get predictions
    top10 = (
        predictions
        .filter(pl.col("customer_id") == cust_id)
        .sort("rank")
        .head(10)
    )
    
    # Check matches
    actual_list = groundtruth.get(cust_id, [])
    if len(actual_list) >= 2:  # Need at least 2 actual purchases
        predicted_items = set(top10['item_id'].to_list())
        actual_items = set(actual_list)
        hits = len(actual_items & predicted_items)
        precision = hits / 10
        
        # Track best case
        if precision > best_precision:
            best_precision = precision
            best_customer = cust_id
            best_top10 = top10
            best_hits = hits
            best_actual = actual_list
        
        # Track worst case
        if precision < worst_precision:
            worst_precision = precision
            worst_customer = cust_id
            worst_top10 = top10
            worst_hits = hits
            worst_actual = actual_list

# Random case
random_customer = random.choice(sample_customers)
random_top10 = predictions.filter(pl.col("customer_id") == random_customer).sort("rank").head(10)
random_actual = groundtruth.get(random_customer, [])
random_hits = len(set(random_top10['item_id'].to_list()) & set(random_actual))

print(f"Analysis complete: Best={best_precision*100:.1f}%, Worst={worst_precision*100:.1f}%, Random customer selected")

# Demo 1: BEST CASE
print("\n" + "="*70)
print("DEMO 1: BEST CASE - Highly Predictable Customer")
print("="*70)

print(f"\nCustomer ID: {best_customer}")
print(f"Profile Type: Brand Loyal / Highly Predictable")

print(f"\nTOP 10 RECOMMENDATIONS:")
for idx, row in enumerate(best_top10.iter_rows(named=True), 1):
    is_match = row['item_id'] in best_actual
    marker = "✓ HIT" if is_match else " "
    print(f"  {idx:2d}. Item {row['item_id']} (score: {row['score']:.3f}) {marker}")

# Evaluation
actual_items = set(best_actual)
predicted_items = set(best_top10['item_id'].to_list())
hits = best_hits

print(f"\nEVALUATION:")
print(f"  Predicted: 10 items")
print(f"  Actual purchases (Jan 2025): {len(actual_items)} items")
print(f"  Matched: {hits} items")
print(f"  Precision@10: {hits/10*100:.1f}%")
print(f"  Status: BEST CASE ⭐")

# Demo 2: WORST CASE
print("\n" + "="*70)
print("DEMO 2: WORST CASE - Unpredictable Customer")
print("="*70)

print(f"\nCustomer ID: {worst_customer}")
print(f"Profile Type: Diverse Shopper / Unpredictable")

print(f"\nTOP 10 RECOMMENDATIONS:")
for idx, row in enumerate(worst_top10.iter_rows(named=True), 1):
    is_match = row['item_id'] in worst_actual
    marker = "✓ HIT" if is_match else " "
    print(f"  {idx:2d}. Item {row['item_id']} (score: {row['score']:.3f}) {marker}")

# Evaluation
actual_items_worst = set(worst_actual)
predicted_items_worst = set(worst_top10['item_id'].to_list())
hits_worst = worst_hits

print(f"\nEVALUATION:")
print(f"  Predicted: 10 items")
print(f"  Actual purchases (Jan 2025): {len(actual_items_worst)} items")
print(f"  Matched: {hits_worst} items")
print(f"  Precision@10: {hits_worst/10*100:.1f}%")
print(f"  Status: WORST CASE ⚠️")

# Demo 3: RANDOM CASE
print("\n" + "="*70)
print("DEMO 3: RANDOM CASE - Typical Customer")
print("="*70)

print(f"\nCustomer ID: {random_customer}")
print(f"Profile Type: Random Sample")

print(f"\nTOP 10 RECOMMENDATIONS:")
for idx, row in enumerate(random_top10.iter_rows(named=True), 1):
    is_match = row['item_id'] in random_actual
    marker = "✓ HIT" if is_match else " "
    print(f"  {idx:2d}. Item {row['item_id']} (score: {row['score']:.3f}) {marker}")

# Evaluation
actual_items_random = set(random_actual)
predicted_items_random = set(random_top10['item_id'].to_list())
hits_random = random_hits

print(f"\nEVALUATION:")
print(f"  Predicted: 10 items")
print(f"  Actual purchases (Jan 2025): {len(actual_items_random)} items")
print(f"  Matched: {hits_random} items")
print(f"  Precision@10: {hits_random/10*100:.1f}%")

if hits_random >= 7:
    print(f"  Status: EXCELLENT (≥70%)")
elif hits_random >= 4:
    print(f"  Status: GOOD (≥40%)")
else:
    print(f"  Status: FAIR (<40%)")

# Feature Importance
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

print("\n[4/4] Extracting feature importance from model...")
try:
    feature_names = model.feature_name()
    feature_importance = model.feature_importance(importance_type='gain')
    
    # Sort by importance
    importance_data = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Most Important Features:")
    total_importance = sum(feature_importance)
    for idx, (feature, importance) in enumerate(importance_data[:10], 1):
        percentage = (importance / total_importance) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {idx:2d}. {feature:30s} {bar} {percentage:5.1f}%")
    
    print(f"\nTotal features: {len(feature_names)}")
except Exception as e:
    print(f"Could not extract feature importance: {e}")

# Overall Summary
print("\n" + "="*70)
print("OVERALL MODEL PERFORMANCE")
print("="*70)

print(f"""
Model: LightGBM (Tuned Hyperparameters)
Training Data: 168M samples, 13 features
Test Set: {format_number(len(groundtruth))} customers

Summary of Demo Cases:
  • Best Case:    {best_hits}/10 matches ({best_hits/10*100:.0f}%) - Customer {best_customer}
  • Worst Case:   {worst_hits}/10 matches ({worst_hits/10*100:.0f}%) - Customer {worst_customer}
  • Random Case:  {random_hits}/10 matches ({random_hits/10*100:.0f}%) - Customer {random_customer}
""")

print("Demo complete! ✓")
print("="*70)