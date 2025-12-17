"""Model training and prediction for recommender system."""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, Literal, List, Dict

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def train_model(
    feature_label_table: pl.LazyFrame,
    feature_columns: List[str],
    label_column: str = "Y",
    model_type: Literal["logistic", "lightgbm"] = "logistic",
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Union[LogisticRegression, "lgb.Booster"]:
    """Train a classification model for purchase prediction.
    
    Supports LogisticRegression (scikit-learn) or LightGBM.
    
    Args:
        feature_label_table: LazyFrame with features and labels.
        feature_columns: List of column names to use as features.
        label_column: Name of the label column (default: "Y").
        model_type: Type of model ("logistic" or "lightgbm").
        model_params: Model-specific parameters (uses defaults if None).
        random_state: Random seed for reproducibility.
        
    Returns:
        Trained model (LogisticRegression or LightGBM Booster).
    """
    # Collect data
    print("Collecting data...")
    df = feature_label_table.collect()
    
    # Prepare features and labels
    X = df.select(feature_columns).to_numpy()
    y = df.select(label_column).to_numpy().ravel()
    
    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive samples: {y.sum()} ({y.mean()*100:.2f}%)")
    
    if model_type == "logistic":
        # Train Logistic Regression
        if model_params is None:
            model_params = {
                "max_iter": 1000,
                "random_state": random_state,
                "solver": "lbfgs",
                "n_jobs": -1,
            }
        
        print("Training Logistic Regression...")
        model = LogisticRegression(**model_params)
        model.fit(X, y)
        
        train_score = model.score(X, y)
        print(f"Training accuracy: {train_score:.4f}")
        
    elif model_type == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
        
        # Default parameters
        if model_params is None:
            model_params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "seed": random_state,
            }
        
        # Create dataset
        train_data = lgb.Dataset(X, label=y, feature_name=feature_columns)
        
        # Train model
        print("Training LightGBM model...")
        model = lgb.train(
            model_params,
            train_data,
            num_boost_round=100,
            callbacks=[
                lgb.log_evaluation(period=10),
            ],
        )
        
        print(f"Training completed")
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model


def predict_and_rank(
    model: Union[LogisticRegression, "lgb.Booster"],
    feature_label_table: pl.LazyFrame,
    feature_columns: List[str],
    user_col: str = "customer_id",
    item_col: str = "item_id",
    top_k: Optional[int] = None,
) -> pl.DataFrame:
    """Generate predictions and rank items for each user.
    
    Args:
        model: Trained model (LogisticRegression or LightGBM Booster).
        feature_label_table: LazyFrame with features for prediction.
        feature_columns: List of feature column names.
        user_col: Name of the user ID column.
        item_col: Name of the item ID column.
        top_k: Return only top K items per user (None for all).
        
    Returns:
        DataFrame with user, item, and prediction score, ranked per user.
    """
    # Collect data
    print("Collecting data for prediction...")
    df = feature_label_table.collect()
    
    # Get features
    X = df.select(feature_columns).to_numpy()
    
    # Predict
    print("Generating predictions...")
    if isinstance(model, LogisticRegression):
        predictions = model.predict_proba(X)[:, 1]  # Probability of positive class
    else:  # LightGBM
        predictions = model.predict(X)
    
    # Add predictions to dataframe
    result = df.select([user_col, item_col]).with_columns(
        pl.Series("score", predictions)
    )
    
    # Rank items per user by score descending
    result = result.sort([user_col, "score"], descending=[False, True])
    
    # Add rank column
    result = result.with_columns(
        pl.col(item_col).rank("dense").over(user_col).alias("rank")
    )
    
    # Filter top K if specified
    if top_k is not None:
        result = result.filter(pl.col("rank") <= top_k)
    
    return result


def evaluate_ranking(
    predictions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    k_values: List[int],
    user_col: str = "customer_id",
    item_col: str = "item_id",
) -> Dict[str, Dict[int, float]]:
    """Evaluate ranking quality using precision@K and NDCG@K.
    
    Efficiently computes metrics using polars operations.
    
    Args:
        predictions: DataFrame with (customer_id, item_id, score, rank).
        ground_truth: DataFrame with positive pairs (customer_id, item_id).
        k_values: List of K values to evaluate.
        user_col: Name of the user ID column.
        item_col: Name of the item ID column.
        
    Returns:
        Dictionary with "precision" and "ndcg" metrics at each K.
    """
    results = {"precision": {}, "ndcg": {}}
    
    # Get ground truth positives
    positives = ground_truth.select([user_col, item_col]).unique()
    
    # Count positives per customer
    positive_counts = (
        positives
        .group_by(user_col)
        .agg(pl.count().alias("num_positives"))
    )
    
    for k in k_values:
        print(f"Computing metrics @{k}...")
        
        # Get top-K predictions
        top_k = predictions.filter(pl.col("rank") <= k)
        
        # Mark hits by joining with ground truth
        hits = (
            top_k
            .join(positives, on=[user_col, item_col], how="inner")
            .select([user_col, item_col, "rank"])
        )
        
        # Count hits per customer
        hit_counts = (
            hits
            .group_by(user_col)
            .agg(pl.count().alias("num_hits"))
        )
        
        # Get all customers who have predictions
        all_customers = top_k.select(user_col).unique()
        
        # Precision@K = hits / K for each customer, then average
        precision_df = (
            all_customers
            .join(hit_counts, on=user_col, how="left")
            .with_columns(
                (pl.col("num_hits").fill_null(0) / k).alias("precision")
            )
        )
        precision_at_k = precision_df["precision"].mean()
        results["precision"][k] = float(precision_at_k)
        
        # NDCG@K computation
        # For each customer, compute DCG and IDCG
        
        # DCG: sum(hit / log2(rank + 1)) for hits
        dcg_df = (
            all_customers
            .join(
                hits.with_columns(
                    (1.0 / (pl.col("rank") + 1).log(2)).alias("dcg_contribution")
                ),
                on=user_col,
                how="left"
            )
            .group_by(user_col)
            .agg(pl.col("dcg_contribution").sum().fill_null(0).alias("dcg"))
        )
        
        # IDCG: sum(1 / log2(r + 1)) for r=1..min(num_positives, k)
        # Precompute ideal gains for ranks 1..k
        ideal_gains = [1.0 / np.log2(r + 1) for r in range(1, k + 1)]
        
        def compute_idcg(num_pos: int) -> float:
            relevant = min(num_pos, k)
            return sum(ideal_gains[:relevant]) if relevant > 0 else 0.0
        
        # Join with positive counts to compute IDCG
        ndcg_df = (
            all_customers
            .join(positive_counts, on=user_col, how="left")
            .with_columns(
                pl.col("num_positives").fill_null(0)
            )
            .join(dcg_df, on=user_col, how="left")
            .with_columns(
                pl.col("dcg").fill_null(0)
            )
        )
        
        # Compute IDCG using numpy for efficiency
        num_positives_arr = ndcg_df["num_positives"].to_numpy()
        idcg_arr = np.array([compute_idcg(int(n)) for n in num_positives_arr])
        
        # NDCG = DCG / IDCG (handle division by zero)
        dcg_arr = ndcg_df["dcg"].to_numpy()
        ndcg_arr = np.where(idcg_arr > 0, dcg_arr / idcg_arr, 0.0)
        
        ndcg_at_k = float(np.mean(ndcg_arr))
        results["ndcg"][k] = ndcg_at_k
        
        print(f"  Precision@{k}: {precision_at_k:.4f}")
        print(f"  NDCG@{k}: {ndcg_at_k:.4f}")
    
    return results


def save_model(model: Union[LogisticRegression, "lgb.Booster"], path: Union[Path, str]) -> None:
    """Save trained model to disk.
    
    Args:
        model: Trained model (LogisticRegression or LightGBM).
        path: Path to save the model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Union[Path, str]) -> Union[LogisticRegression, "lgb.Booster"]:
    """Load trained model from disk.
    
    Args:
        path: Path to the saved model.
        
    Returns:
        Loaded model (LogisticRegression or LightGBM).
    """
    print(f"Loading model from {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def get_feature_importance(
    model: Union[LogisticRegression, "lgb.Booster"],
    feature_names: List[str],
    importance_type: str = "gain",
) -> pl.DataFrame:
    """Get feature importance from trained model.
    
    Args:
        model: Trained model (LogisticRegression or LightGBM).
        feature_names: List of feature names.
        importance_type: Type of importance ('gain' or 'split') for LightGBM.
        
    Returns:
        DataFrame with feature names and importance scores.
    """
    if isinstance(model, LogisticRegression):
        # For logistic regression, use absolute coefficients
        importance = np.abs(model.coef_[0])
    else:  # LightGBM
        importance = model.feature_importance(importance_type=importance_type)
    
    df = pl.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort("importance", descending=True)
    
    return df



