"""
Compare results from different feature sets
Reads metrics JSON files and generates comparison table
"""

import json
import os
from glob import glob
from datetime import datetime

print("="*80)
print("FEATURE COMPARISON - LIGHTGBM MODELS")
print("="*80)

# Find all metrics files
metrics_pattern = "outputs/metrics_lightgbm_*.json"
metrics_files = glob(metrics_pattern)

if not metrics_files:
    print(f"\nNo metrics files found matching: {metrics_pattern}")
    print("Please run the training scripts first:")
    print("  - python train_lightgbm_3features.py")
    print("  - python train_lightgbm_5features.py")
    print("  - python train_lightgbm_9features.py")
    print("  - python train_lightgbm_without_history.py")
    exit(1)

# Group by feature count
results = {}

for file in metrics_files:
    # Extract feature count from filename
    if "without_history" in file:
        feature_count = 10
        feature_set = "10 features (X4-X13, no history)"
    elif "3features" in file:
        feature_count = 3
        feature_set = "3 features (baseline)"
    elif "5features" in file:
        feature_count = 5
        feature_set = "5 features (+recency/freq)"
    elif "9features" in file:
        feature_count = 9
        feature_set = "9 features (+monetary/brand)"
    elif "tuned" in file:
        feature_count = 13
        feature_set = "13 features (X1-X13, with history)"
    else:
        continue
    
    # Load metrics
    with open(file, 'r') as f:
        metrics = json.load(f)
    
    # Store results
    if feature_count not in results or "tuned" in file:
        results[feature_count] = {
            'feature_set': feature_set,
            'metrics': metrics,
            'file': file
        }

# Sort by feature count
sorted_results = sorted(results.items())

print(f"\nFound {len(sorted_results)} models to compare:\n")

# Print comparison table
print("="*80)
print("METRICS COMPARISON TABLE")
print("="*80)

# Header
print(f"\n{'Feature Set':<30} {'P@5':<8} {'P@10':<8} {'P@20':<8} {'R@10':<8} {'R@20':<8} {'NDCG@10':<8} {'MAP@10':<8}")
print("-"*100)

# Rows
for feature_count, data in sorted_results:
    metrics = data['metrics']
    feature_set = data['feature_set']
    
    # Extract values - handle both string and int keys
    precision_dict = metrics.get('precision', {})
    recall_dict = metrics.get('recall', {})
    ndcg_dict = metrics.get('ndcg', {})
    map_dict = metrics.get('map', {})
    
    # Try both string and int keys
    p5 = precision_dict.get('5', precision_dict.get(5, 0))
    p10 = precision_dict.get('10', precision_dict.get(10, 0))
    p20 = precision_dict.get('20', precision_dict.get(20, 0))
    r10 = recall_dict.get('10', recall_dict.get(10, 0))
    r20 = recall_dict.get('20', recall_dict.get(20, 0))
    ndcg10 = ndcg_dict.get('10', ndcg_dict.get(10, 0))
    map10 = map_dict.get('10', map_dict.get(10, 0))
    
    print(f"{feature_set:<30} {p5:<8.4f} {p10:<8.4f} {p20:<8.4f} {r10:<8.4f} {r20:<8.4f} {ndcg10:<8.4f} {map10:<8.4f}")

print("\n" + "="*80)
print("IMPROVEMENT ANALYSIS")
print("="*80)

# Calculate improvements
if len(sorted_results) >= 2:
    baseline = sorted_results[0][1]['metrics']
    
    print(f"\nBaseline: {sorted_results[0][1]['feature_set']}")
    print("-"*80)
    
    for i in range(1, len(sorted_results)):
        feature_count, data = sorted_results[i]
        metrics = data['metrics']
        feature_set = data['feature_set']
        
        # Calculate % improvement for P@10
        baseline_p10 = baseline.get('precision', {}).get(10, 0)
        current_p10 = metrics.get('precision', {}).get(10, 0)
        
        if baseline_p10 > 0:
            improvement_p10 = ((current_p10 - baseline_p10) / baseline_p10) * 100
        else:
            improvement_p10 = 0
        
        # Calculate % improvement for R@10
        baseline_r10 = baseline.get('recall', {}).get(10, 0)
        current_r10 = metrics.get('recall', {}).get(10, 0)
        
        if baseline_r10 > 0:
            improvement_r10 = ((current_r10 - baseline_r10) / baseline_r10) * 100
        else:
            improvement_r10 = 0
        
        # Calculate % improvement for NDCG@10
        baseline_ndcg10 = baseline.get('ndcg', {}).get(10, 0)
        current_ndcg10 = metrics.get('ndcg', {}).get(10, 0)
        
        if baseline_ndcg10 > 0:
            improvement_ndcg10 = ((current_ndcg10 - baseline_ndcg10) / baseline_ndcg10) * 100
        else:
            improvement_ndcg10 = 0
        
        print(f"\n{feature_set}:")
        print(f"  Added features: {feature_count - sorted_results[0][0]}")
        print(f"  Precision@10:  {current_p10:.4f} ({improvement_p10:+.2f}%)")
        print(f"  Recall@10:     {current_r10:.4f} ({improvement_r10:+.2f}%)")
        print(f"  NDCG@10:       {current_ndcg10:.4f} ({improvement_ndcg10:+.2f}%)")

# Save summary
print("\n" + "="*80)
print("SAVING SUMMARY")
print("="*80)

summary = {
    'comparison_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'models_compared': len(sorted_results),
    'results': []
}

for feature_count, data in sorted_results:
    summary['results'].append({
        'feature_count': feature_count,
        'feature_set': data['feature_set'],
        'metrics': data['metrics'],
        'metrics_file': data['file']
    })

output_file = f"outputs/feature_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nSummary saved to: {output_file}")

print("\n" + "="*80)
print("DONE - COMPARISON COMPLETE")
print("="*80)
