# CS116 - Product Recommendation System

A machine learning recommender system for predicting customer purchases using multiple classification models (LightGBM, XGBoost, Random Forest, Logistic Regression).

## Project Structure

```
neSemi/
├── src/
│   └── recommender/
│       ├── __init__.py
│       ├── data.py          # Data loading and schemas
│       ├── features.py      # Feature engineering
│       ├── candidates.py    # Candidate generation
│       ├── metrics.py       # Evaluation metrics (Precision@K, Recall@K, NDCG@K)
│       ├── train.py         # Model training and prediction
│       ├── config.py        # Configuration settings
│       └── utils.py         # Utility functions
├── notebooks/
│   ├── eda_analysis.ipynb          # Exploratory data analysis
│   └── analyze_results.ipynb       # Model results analysis
├── outputs/
│   ├── models/                     # Trained model files (.pkl)
│   ├── predictions/                # Prediction results (.parquet)
│   ├── features/                   # Feature caches
│   └── metrics_*.json              # Model evaluation metrics
├── train_all_models.py             # Train all 4 models
├── train_lightgbm_parameter.py     # LightGBM hyperparameter tuning
├── demo_presentation.py            # Live demo with sample predictions
├── optimize_submission.py          # Optimize predictions for submission
├── requirements.txt
└── README.md
```

## Data Schema

### Transactions
- `customer_id`: int - Customer identifier
- `item_id`: int - Product identifier
- `created_at`: datetime - Purchase timestamp
- `order_id`: int - Order identifier (optional)

### Items
- `item_id`: int - Product identifier
- `brand`: str - Product brand
- `age_group`: str - Target age group
- `category`: str - Product category

### Users
- `customer_id`: int - Customer identifier
- `date_of_birth`: date - Customer birth date

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- polars >= 0.20.0 (efficient data processing)
- scikit-learn >= 1.3.0 (Logistic Regression, Random Forest)
- lightgbm >= 4.0.0 (LightGBM model)
- xgboost (XGBoost model)
- numpy, pandas, pyarrow, matplotlib, jupyter

## Usage

### 1. Train All Models
Train all 4 models (Logistic Regression, LightGBM, XGBoost, Random Forest):

```bash
python train_all_models.py
```

**Output:**
- Model files: `outputs/models/model_*.pkl`
- Predictions: `outputs/predictions/predictions_*.parquet`
- Metrics: `outputs/metrics_*.json`

### 2. Hyperparameter Tuning (LightGBM)
Fine-tune LightGBM parameters:

```bash
python train_lightgbm_parameter.py
```

### 3. Live Demo
See sample predictions with customer profiles:

```bash
python demo_presentation.py
```

Shows:
- Customer purchase history
- Top 10 product recommendations
- Evaluation metrics (Precision@10, Recall@10, NDCG@10)

### 4. Optimize Submission
Generate optimized submission file (8-90MB):

```bash
python optimize_submission.py
```

**Output:** `outputs/submission_lightgbm_optimized.json`

## Models

The system supports 4 classification models:

1. **Logistic Regression** - Baseline model, fast training
2. **Random Forest** - Ensemble model with decision trees
3. **XGBoost** - Gradient boosting with regularization
4. **LightGBM** - Fast gradient boosting (best performance)

## Features

**Core Features:**
- Polars LazyFrame for efficient large-scale data processing
- Time-based feature engineering (historical vs recent purchases)
- Candidate generation strategies (item-based, user-based)
- Comprehensive evaluation metrics (Precision@K, Recall@K, NDCG@K, F1@K)

**Feature Engineering:**
- User purchase frequency and recency
- Item popularity and purchase patterns
- Brand loyalty metrics
- Category preferences
- Time-based aggregations

**Output Files:**
- `outputs/models/` - Trained model files (pickle format)
- `outputs/predictions/` - Customer-item predictions with scores
- `outputs/metrics_*.json` - Model evaluation results
- `outputs/submission_*.json` - Submission-ready predictions

## Model Performance

**Latest Results (Dec 21, 2025):**

Best model: **LightGBM (tuned)**

**Public Test Score: 6.89%** ⭐

Internal validation metrics (new groundtruth - 644,970 customers):
- Precision@10: 4.38%
- Recall@10: 12.30%
- NDCG@10: 9.62%
- F1@10: 6.46%

Submission details:
- Customers: 100,000 (top customers by score)
- Items per customer: 10
- File size: 14.33 MB
- Model: LightGBM trained on 2024 data (Jan-Nov)

## Development

**Notebooks:**
- [eda_analysis.ipynb](notebooks/eda_analysis.ipynb) - Data exploration and insights
- [analyze_results.ipynb](notebooks/analyze_results.ipynb) - Model comparison and analysis

**Key Scripts:**
- `train_all_models.py` - Batch training all models
- `train_lightgbm_parameter.py` - Hyperparameter optimization
- `demo_presentation.py` - Interactive demo for presentations
- `optimize_submission.py` - Prepare submission file
