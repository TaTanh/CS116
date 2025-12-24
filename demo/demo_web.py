"""
DEMO WEB APP - Product Recommendation System
Run: python demo_web.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify
import pickle
import polars as pl
import os
import random

app = Flask(__name__)

# Global variables to store loaded data
model = None
predictions = None
groundtruth = None
best_case = None
worst_case = None
random_case = None
feature_importance_data = None

def format_number(n):
    """Format number with thousand separators"""
    return f"{n:,}"

def load_data():
    """Load model, predictions and groundtruth"""
    global model, predictions, groundtruth
    
    print("Loading model and data...")
    model_path = "../outputs/models/model_lightgbm_tuned_20251221_103746.pkl"
    predictions_path = "../outputs/predictions/predictions_lightgbm_tuned_20251221_103746.parquet"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = pickle.load(open(model_path, "rb"))
    predictions = pl.read_parquet(predictions_path)
    
    with open("../groundtruth.pkl", "rb") as f:
        groundtruth = pickle.load(f)
    
    print(f"✓ Model loaded")
    print(f"✓ Predictions loaded: {format_number(predictions.shape[0])} rows")
    print(f"✓ Groundtruth loaded: {format_number(len(groundtruth))} customers")

def analyze_cases():
    """Find best, worst, and random cases"""
    global best_case, worst_case, sample_customers, feature_importance_data
    
    print("Analyzing demo cases...")
    
    # Sample customers - store globally for random refresh
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
    
    for cust_id in sample_customers:
        top10 = (
            predictions
            .filter(pl.col("customer_id") == cust_id)
            .sort("rank")
            .head(10)
        )
        
        actual_list = groundtruth.get(cust_id, [])
        if len(actual_list) >= 2:
            predicted_items = set(top10['item_id'].to_list())
            actual_items = set(actual_list)
            hits = len(actual_items & predicted_items)
            precision = hits / 10
            
            if precision > best_precision:
                best_precision = precision
                best_case = {
                    'customer_id': cust_id,
                    'top10': top10.to_dicts(),
                    'actual': actual_list,
                    'hits': hits,
                    'precision': precision
                }
            
            if precision < worst_precision:
                worst_precision = precision
                worst_case = {
                    'customer_id': cust_id,
                    'top10': top10.to_dicts(),
                    'actual': actual_list,
                    'hits': hits,
                    'precision': precision
                }
    
    print(f"✓ Best case: {best_precision*100:.1f}%")
    print(f"✓ Worst case: {worst_precision*100:.1f}%")
    print(f"✓ Sample customers ready for random selection")
    
    # Feature importance
    try:
        feature_names = model.feature_name()
        feature_importance = model.feature_importance(importance_type='gain')
        total_importance = sum(feature_importance)
        
        feature_importance_data = [
            {
                'name': name,
                'importance': float(imp),
                'percentage': float((imp / total_importance) * 100)
            }
            for name, imp in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)
        ]
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        feature_importance_data = []

@app.route('/')
def index():
    """Main demo page"""
    return render_template('demo.html')

@app.route('/api/overview')
def api_overview():
    """Get overview statistics"""
    return jsonify({
        'total_predictions': predictions.shape[0],
        'total_customers': len(groundtruth),
        'model_name': 'LightGBM (Tuned)',
        'training_samples': '168M',
        'num_features': len(model.feature_name()) if model else 13
    })

@app.route('/api/cases')
def api_cases():
    """Get all demo cases"""
    return jsonify({
        'best': best_case,
        'worst': worst_case,
        'random': get_random_case()  # Generate new random case each time
    })

def get_random_case():
    """Generate a new random case"""
    random_customer = random.choice(sample_customers)
    random_top10 = predictions.filter(pl.col("customer_id") == random_customer).sort("rank").head(10)
    random_actual = groundtruth.get(random_customer, [])
    random_hits = len(set(random_top10['item_id'].to_list()) & set(random_actual))
    
    return {
        'customer_id': random_customer,
        'top10': random_top10.to_dicts(),
        'actual': random_actual,
        'hits': random_hits,
        'precision': random_hits / 10
    }

@app.route('/api/feature_importance')
def api_feature_importance():
    """Get feature importance data"""
    return jsonify(feature_importance_data)

if __name__ == '__main__':
    load_data()
    analyze_cases()
    print("\n" + "="*70)
    print("🚀 Starting web server...")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
