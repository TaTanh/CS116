"""
Convert final_groundtruth.pkl (DataFrame) to groundtruth.pkl (dict format)
"""

import pickle
import shutil
from datetime import datetime

print("="*70)
print("CONVERT FINAL_GROUNDTRUTH.PKL TO DICT FORMAT")
print("="*70)

# 1. Backup old groundtruth.pkl
print("\n[1] Backing up old groundtruth.pkl...")
backup_name = f"groundtruth_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
shutil.copy("groundtruth.pkl", backup_name)
print(f"Backed up to: {backup_name}")

# 2. Load final_groundtruth.pkl
print("\n[2] Loading final_groundtruth.pkl...")
with open("final_groundtruth.pkl", "rb") as f:
    final_gt_df = pickle.load(f)

print(f"Type: {type(final_gt_df).__name__}")
print(f"Shape: {final_gt_df.shape}")
print(f"Columns: {list(final_gt_df.columns)}")

# 3. Convert DataFrame to dict
print("\n[3] Converting to dict format...")
import numpy as np

# Group by customer_id and aggregate item_id into lists
groundtruth_dict = {}
for _, row in final_gt_df.iterrows():
    customer_id = int(row['customer_id'])
    item_ids = row['item_id']
    
    # item_id is numpy array of strings
    if isinstance(item_ids, np.ndarray):
        # Convert numpy array to list of strings
        groundtruth_dict[customer_id] = item_ids.tolist()
    elif isinstance(item_ids, (list, tuple)):
        groundtruth_dict[customer_id] = [str(item) for item in item_ids]
    else:
        groundtruth_dict[customer_id] = [str(item_ids)]

print(f"Converted to dict with {len(groundtruth_dict):,} customers")
print(f"Sample: {list(groundtruth_dict.items())[0]}")

# 4. Save as new groundtruth.pkl
print("\n[4] Saving as groundtruth.pkl...")
with open("groundtruth.pkl", "wb") as f:
    pickle.dump(groundtruth_dict, f)

print("Saved successfully!")

# 5. Verify
print("\n[5] Verifying new file...")
with open("groundtruth.pkl", "rb") as f:
    verify_gt = pickle.load(f)

print(f"Type: {type(verify_gt)}")
print(f"Customers: {len(verify_gt):,}")
print(f"Sample customer {list(verify_gt.keys())[0]}: {verify_gt[list(verify_gt.keys())[0]][:5]}")

print("\n" + "="*70)
print("CONVERSION COMPLETED!")
print("="*70)
print(f"Old file backed up: {backup_name}")
print(f"New groundtruth.pkl: {len(verify_gt):,} customers")
print(f"Increase: +{len(verify_gt) - 391900:,} customers")
