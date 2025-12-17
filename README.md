# CS116
# Time-Based Purchase Prediction Recommender

A machine learning recommender system for predicting future customer purchases based on historical transaction data.

## Project Structure

```
neSemi/
├── src/
│   └── recommender/
│       ├── __init__.py
│       ├── data.py          # Data schemas and loading utilities
│       ├── features.py      # Feature engineering
│       ├── candidates.py    # Candidate generation
│       ├── metrics.py       # Evaluation metrics
│       └── train.py         # Model training and prediction
├── main.py                  # CLI entry point
├── requirements.txt
└── README.md
```

## Data Schema

### Transactions
- `customer_id`: int
- `item_id`: int
- `created_at`: datetime
- `order_id`: int (optional)

### Items
- `item_id`: int
- `brand`: str
- `age_group`: str
- `category`: str

### Users
- `customer_id`: int
- `date_of_birth`: date

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Build features
python main.py build-features --transactions-path data/transactions.parquet --items-path data/items.parquet --users-path data/users.parquet --output-path data/features.parquet

# Train model
python main.py train --features-path data/features.parquet --output-path models/model.pkl

# Generate predictions
python main.py predict --model-path models/model.pkl --features-path data/features.parquet --output-path predictions.parquet
```

## Features

- Polars LazyFrame for efficient data processing
- Time-based feature engineering
- Candidate generation strategies
- Standard recommendation metrics (Precision@K, NDCG@K)
- Modular design for easy experimentation
