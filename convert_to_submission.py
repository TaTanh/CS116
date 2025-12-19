"""
Convert model predictions to JSON format for submission.
Format: {"customer_id": ["item_1", "item_2", ...]}
"""

import pickle
import json
import polars as pl
from pathlib import Path

print("="*70)
print("CONVERT PREDICTIONS TO SUBMISSION FORMAT")
print("="*70)

# 1. Load groundtruth to understand the format
print("\n[1] Loading groundtruth.pkl...")
with open("groundtruth.pkl", "rb") as f:
    groundtruth = pickle.load(f)

print(f"Groundtruth type: {type(groundtruth)}")
print(f"Groundtruth shape/size: {len(groundtruth) if hasattr(groundtruth, '__len__') else 'N/A'}")
print(f"\nFirst few entries:")
if isinstance(groundtruth, dict):
    for i, (k, v) in enumerate(list(groundtruth.items())[:3]):
        print(f"  {k}: {v}")
elif isinstance(groundtruth, pl.DataFrame):
    print(groundtruth.head(5))
else:
    print(groundtruth[:5] if hasattr(groundtruth, '__getitem__') else groundtruth)

# 2. Load predictions (using LightGBM - best model)
print("\n[2] Loading predictions from LightGBM (best model)...")
predictions_file = "outputs/predictions/predictions_lightgbm_20251219_190733.parquet"
predictions_df = pl.read_parquet(predictions_file)
print(f"Predictions shape: {predictions_df.shape}")
print(f"Columns: {predictions_df.columns}")
print(predictions_df.head(10))

# 3. Convert to submission format
print("\n[3] Converting to submission format...")
# Group by customer_id and collect top-K items
submission_dict = {}

for customer_id in predictions_df["customer_id"].unique():
    # Get items for this customer, sorted by rank
    customer_items = (
        predictions_df
        .filter(pl.col("customer_id") == customer_id)
        .sort("rank")
        .select("item_id")
        .to_series()
        .to_list()
    )
    
    # Convert to format: "cus_XXX"
    customer_key = f"cus_{customer_id:03d}" if isinstance(customer_id, int) else str(customer_id)
    
    # Convert items to format: "item_XXX"
    item_list = [f"item_{item_id}" if isinstance(item_id, int) else str(item_id) 
                 for item_id in customer_items]
    
    submission_dict[customer_key] = item_list

print(f"Total customers: {len(submission_dict)}")
print(f"\nFirst 3 customers:")
for i, (k, v) in enumerate(list(submission_dict.items())[:3]):
    print(f"  {k}: {v[:5]}... ({len(v)} items)")

# 4. Save to JSON
output_file = "outputs/submission_lightgbm.json"
print(f"\n[4] Saving to {output_file}...")
with open(output_file, "w") as f:
    json.dump(submission_dict, f, indent=2)

print(f"✓ Submission file created: {output_file}")
print(f"✓ Best model: LightGBM (Precision@10: 0.0408, NDCG@10: 0.1182)")
print(f"✓ Ready to upload to teacher's website!")

print("\n" + "="*70)
print("DONE!")
print("="*70)
