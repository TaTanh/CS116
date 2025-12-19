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

# 2. Load predictions
print("\n[2] Loading predictions...")
predictions_file = "outputs/predictions/predictions_lightgbm_20251219_213020.parquet"
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

# 4. Convert to submission format (compact)
print("\n[4] Converting to compact JSON format...")
submission_dict = {}

for customer_id in predictions_filtered["customer_id"].unique():
    # Only include if customer is in groundtruth
    if customer_id not in valid_customers:
        continue
    
    # Get top-K items for this customer
    customer_items = (
        predictions_filtered
        .filter(pl.col("customer_id") == customer_id)
        .sort("rank")
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
print(f"✓ File size: {file_size_mb:.2f} MB")

if file_size_mb > 90:
    print(f"⚠️  Still too large! Consider reducing top-K from 20 to 10-15")
else:
    print(f"✓ File size OK for upload!")

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"\nOptimized file: {output_file}")
print(f"Size: {file_size_mb:.2f} MB")
