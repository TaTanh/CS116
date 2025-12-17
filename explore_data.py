"""
Quick script to explore what parquet files are available in the dataset directory.
Run this first to see what data you have.
"""

from src.recommender import explore_dataset, DATA_DIR

print(f"\n{'='*70}")
print(f"  DATASET EXPLORER")
print(f"{'='*70}\n")

explore_dataset()

print(f"\n{'='*70}")
print("DONE! You can now use these files in your analysis.")
print(f"{'='*70}\n")
print("Tips:")
print("  - Use load_transactions(), load_items(), load_users() for default files")
print("  - Use load_any_parquet('filename.parquet') for other files")
print("  - Run 'python example_workflow.py' to test the full pipeline")
