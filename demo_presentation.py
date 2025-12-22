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

def format_number(n):
    """Format number with thousand separators"""
    return f"{n:,}"

print("="*70)
print("PRODUCT RECOMMENDATION SYSTEM - LIVE DEMO")
print("="*70)

# Load model and predictions
print("\n[1/4] Loading model and predictions...")
model_path = "outputs/models/model_lightgbm_tuned_20251221_103746.pkl"
predictions_path = "outputs/predictions/predictions_lightgbm_tuned_20251221_103746.parquet"

if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    exit(1)

model = pickle.load(open(model_path, "rb"))
predictions = pl.read_parquet(predictions_path)
print("Model loaded successfully")
print(f"Predictions loaded: {format_number(predictions.shape[0])} rows")

# Load groundtruth
print("\n[2/4] Loading groundtruth...")
with open("groundtruth.pkl", "rb") as f:
    groundtruth = pickle.load(f)
print(f"Groundtruth loaded: {format_number(len(groundtruth))} customers")

# Demo 1: GOOD CASE - Find customer with actual good matches
print("\n" + "="*70)
print("DEMO 1: EXCELLENT PREDICTION (Brand Loyal Customer)")
print("="*70)

# Find customers with good matches
print("\n[3/4] Finding best demo cases...")
best_precision = 0
good_customer = None
good_top10 = None
good_hits = 0
good_actual = []

# Sample 500 customers to find good case
sample_customers = (
    predictions
    .select("customer_id")
    .unique()
    .filter(pl.col("customer_id").is_in(list(groundtruth.keys())))
    .head(500)
    .to_series()
    .to_list()
)

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
    if len(actual_list) >= 3:  # Need enough actual purchases
        predicted_items = set(top10['item_id'].to_list())
        actual_items = set(actual_list)
        hits = len(actual_items & predicted_items)
        precision = hits / 10
        
        if precision > best_precision:
            best_precision = precision
            good_customer = cust_id
            good_top10 = top10
            good_hits = hits
            good_actual = actual_list

if good_customer is None:
    print("No good cases found in sample, using first customer")
    good_customer = sample_customers[0]
    good_top10 = predictions.filter(pl.col("customer_id") == good_customer).sort("rank").head(10)
    good_actual = groundtruth.get(good_customer, [])
    good_hits = len(set(good_top10['item_id'].to_list()) & set(good_actual))

print(f"\nCustomer ID: {good_customer}")
print(f"Profile Type: Brand Loyal / High Precision")

print(f"\nTOP 10 RECOMMENDATIONS:")
for idx, row in enumerate(good_top10.iter_rows(named=True), 1):
    is_match = row['item_id'] in good_actual
    marker = "✓" if is_match else " "
    print(f"  {idx:2d}. Item {row['item_id']} (score: {row['score']:.3f}) {marker}")

# Evaluation
actual_items = set(good_actual)
predicted_items = set(good_top10['item_id'].to_list())
hits = good_hits

print(f"\nEVALUATION:")
print(f"  Predicted: 10 items")
print(f"  Actual purchases (Jan 2025): {len(actual_items)} items")
print(f"  Matched: {hits} items")
print(f"  Precision@10: {hits/10*100:.1f}%")

if hits >= 7:
    print(f"  Status: EXCELLENT (≥70%)")
elif hits >= 4:
    print(f"  Status: GOOD (≥40%)")
else:
    print(f"  Status: FAIR (<40%)")

# Demo 2: POOR CASE - Diverse Shopper
print("\n" + "="*70)
print("DEMO 2: CHALLENGING PREDICTION (Diverse Shopper)")
print("="*70)

# Find a customer with low score predictions
poor_customer = (
    predictions
    .group_by("customer_id")
    .agg(pl.col("score").mean().alias("avg_score"))
    .sort("avg_score", descending=False)
    .filter(pl.col("customer_id").is_in(list(groundtruth.keys())))
    .head(100)
    .tail(1)
    .select("customer_id")
    .item()
)

print(f"\nCustomer ID: {poor_customer}")
print(f"Profile Type: Diverse Shopper / Unpredictable")

# Get top 10 recommendations
top10_poor = (
    predictions
    .filter(pl.col("customer_id") == poor_customer)
    .sort("rank")
    .head(10)
)

print(f"\nTOP 10 RECOMMENDATIONS:")
for idx, row in enumerate(top10_poor.iter_rows(named=True), 1):
    print(f"  {idx:2d}. Item {row['item_id']} (confidence: {row['score']:.3f})")

# Evaluation
actual_items_list_poor = groundtruth.get(poor_customer, [])
if actual_items_list_poor:
    actual_items_poor = set(actual_items_list_poor)
    predicted_items_poor = set(top10_poor['item_id'].to_list())
    hits_poor = len(actual_items_poor & predicted_items_poor)
else:
    actual_items_poor = set()
    hits_poor = 0

print(f"\nEVALUATION:")
print(f"  Predicted: 10 items")
print(f"  Actual purchases (Jan 2025): {len(actual_items_poor)} items")
print(f"  Matched: {hits_poor} items")
print(f"  Precision@10: {hits_poor/10*100:.1f}%")

if hits_poor >= 7:
    print(f"  Status: EXCELLENT (≥70%)")
elif hits_poor >= 4:
    print(f"  Status: GOOD (≥40%)")
else:
    print(f"  Status: CHALLENGING (<40%)")

# Overall Summary
print("\n" + "="*70)
print("OVERALL MODEL PERFORMANCE")
print("="*70)

print(f"""
Model: LightGBM (Tuned Hyperparameters)
Training Data: 168M samples, 13 features
Test Set: {format_number(len(groundtruth))} customers
""")