"""
Optimize submission JSON for upload (target: 8-90MB)
Only include customers in groundtruth and remove JSON formatting
"""

import pickle
import json
import polars as pl

print("="*70)
print("OPTIMIZE SUBMISSION JSON")
print("="*70)

# 1. Load groundtruth to get valid customers
print("\n[1] Loading groundtruth.pkl...")
with open("groundtruth.pkl", "rb") as f:
    groundtruth = pickle.load(f)

valid_customers = set(groundtruth.keys())
print(f"Valid customers in groundtruth: {len(valid_customers):,}")

# 2. Load predictions (LightGBM with new groundtruth - 6.89%)
print("\n[2] Loading predictions...")
predictions_file = "outputs/predictions/predictions_new_groundtruth_20251221_222506.parquet"
predictions_df = pl.read_parquet(predictions_file)
print(f"Total predictions: {predictions_df.shape[0]:,}")
print(f"Total customers: {predictions_df['customer_id'].n_unique():,}")

# 3. Filter predictions to only valid customers
print("\n[3] Filtering to groundtruth customers only...")
predictions_filtered = predictions_df.filter(
    pl.col("customer_id").is_in(list(valid_customers))
)
print(f"Filtered predictions: {predictions_filtered.shape[0]:,}")
print(f"Filtered customers: {predictions_filtered['customer_id'].n_unique():,}")

# 3.5. Select TOP customers by average score (to reduce file size)
print("\n[3.5] Selecting TOP 100K customers by avg score...")
customer_avg_scores = (
    predictions_filtered
    .group_by("customer_id")
    .agg(pl.col("score").mean().alias("avg_score"))
    .sort("avg_score", descending=True)
)
top_n = min(100000, customer_avg_scores.shape[0])  # Reduced from 190K to 100K
top_customers = set(customer_avg_scores.head(top_n)["customer_id"].to_list())
print(f"Selected top {len(top_customers):,} customers")

predictions_filtered = predictions_filtered.filter(
    pl.col("customer_id").is_in(list(top_customers))
)

# 4. Convert to submission format (compact - TOP 10 items)
print("\n[4] Converting to compact JSON format (TOP 10 items)...")
submission_dict = {}

for customer_id in predictions_filtered["customer_id"].unique():
    # Only include if customer is in groundtruth
    if customer_id not in valid_customers:
        continue
    
    # Get top-10 items for this customer
    customer_items = (
        predictions_filtered
        .filter(pl.col("customer_id") == customer_id)
        .sort("rank")
        .head(10)  # TOP 10
        .select("item_id")
        .to_series()
        .to_list()
    )
    
    # Use customer_id as key directly (no "cus_" prefix to save space)
    submission_dict[str(customer_id)] = customer_items

print(f"Submission customers: {len(submission_dict):,}")

# 5. Save with NO indent (compact format)
output_file = "outputs/submission_lightgbm_optimized.json"
print(f"\n[5] Saving compact JSON to {output_file}...")
with open(output_file, "w") as f:
    json.dump(submission_dict, f, separators=(',', ':'))  # No spaces, no indent

# Check file size
import os
file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"File size: {file_size_mb:.2f} MB")

if file_size_mb > 90:
    print(f"File too large. Consider reducing customers or items per customer")
elif file_size_mb < 8:
    print(f"File might be too small. Consider increasing customers")
else:
    print(f"File size OK for upload!")

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"\nOptimized file: {output_file}")
print(f"Size: {file_size_mb:.2f} MB")
