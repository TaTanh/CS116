"""Evaluation metrics for recommender system."""

from typing import Sequence, Dict

import numpy as np
import polars as pl


def precision_at_k(
    predictions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    k: int,
    user_col: str = "customer_id",
    item_col: str = "item_id",
    score_col: str = "score",
) -> float:
    """Calculate Precision@K metric.
    
    Precision@K measures the proportion of recommended items in the top-K
    that are relevant (i.e., actually purchased).
    
    Args:
        predictions: DataFrame with predictions (user, item, score).
        ground_truth: DataFrame with actual purchases (user, item).
        k: Number of top recommendations to consider.
        user_col: Name of the user ID column.
        item_col: Name of the item ID column.
        score_col: Name of the score/ranking column.
        
    Returns:
        Average precision@K across all users.
    """
    # Get top-K predictions per user
    top_k_preds = (
        predictions
        .sort([user_col, score_col], descending=[False, True])
        .group_by(user_col)
        .agg([
            pl.col(item_col).head(k).alias("predicted_items"),
        ])
    )
    
    # Get actual purchased items per user
    actual = (
        ground_truth
        .group_by(user_col)
        .agg([
            pl.col(item_col).alias("actual_items"),
        ])
    )
    
    # Join predictions with ground truth
    joined = top_k_preds.join(actual, on=user_col, how="inner")
    
    # Calculate precision for each user
    precisions = []
    for row in joined.iter_rows(named=True):
        predicted_set = set(row["predicted_items"])
        actual_set = set(row["actual_items"])
        
        hits = len(predicted_set & actual_set)
        precision = hits / k if k > 0 else 0.0
        precisions.append(precision)
    
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(
    predictions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    k: int,
    user_col: str = "customer_id",
    item_col: str = "item_id",
    score_col: str = "score",
) -> float:
    """Calculate Recall@K metric.
    
    Recall@K measures the proportion of relevant items that appear in
    the top-K recommendations.
    
    Args:
        predictions: DataFrame with predictions (user, item, score).
        ground_truth: DataFrame with actual purchases (user, item).
        k: Number of top recommendations to consider.
        user_col: Name of the user ID column.
        item_col: Name of the item ID column.
        score_col: Name of the score/ranking column.
        
    Returns:
        Average recall@K across all users.
    """
    # Get top-K predictions per user
    top_k_preds = (
        predictions
        .sort([user_col, score_col], descending=[False, True])
        .group_by(user_col)
        .agg([
            pl.col(item_col).head(k).alias("predicted_items"),
        ])
    )
    
    # Get actual purchased items per user
    actual = (
        ground_truth
        .group_by(user_col)
        .agg([
            pl.col(item_col).alias("actual_items"),
        ])
    )
    
    # Join predictions with ground truth
    joined = top_k_preds.join(actual, on=user_col, how="inner")
    
    # Calculate recall for each user
    recalls = []
    for row in joined.iter_rows(named=True):
        predicted_set = set(row["predicted_items"])
        actual_set = set(row["actual_items"])
        
        hits = len(predicted_set & actual_set)
        total_relevant = len(actual_set)
        recall = hits / total_relevant if total_relevant > 0 else 0.0
        recalls.append(recall)
    
    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(
    predictions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    k: int,
    user_col: str = "customer_id",
    item_col: str = "item_id",
    score_col: str = "score",
) -> float:
    """Calculate Normalized Discounted Cumulative Gain@K metric.
    
    NDCG@K measures ranking quality by giving more weight to relevant
    items appearing earlier in the recommendation list.
    
    Args:
        predictions: DataFrame with predictions (user, item, score).
        ground_truth: DataFrame with actual purchases (user, item).
        k: Number of top recommendations to consider.
        user_col: Name of the user ID column.
        item_col: Name of the item ID column.
        score_col: Name of the score/ranking column.
        
    Returns:
        Average NDCG@K across all users.
    """
    # Get top-K predictions per user
    top_k_preds = (
        predictions
        .sort([user_col, score_col], descending=[False, True])
        .group_by(user_col)
        .agg([
            pl.col(item_col).head(k).alias("predicted_items"),
        ])
    )
    
    # Get actual purchased items per user
    actual = (
        ground_truth
        .group_by(user_col)
        .agg([
            pl.col(item_col).alias("actual_items"),
        ])
    )
    
    # Join predictions with ground truth
    joined = top_k_preds.join(actual, on=user_col, how="inner")
    
    # Calculate NDCG for each user
    ndcgs = []
    for row in joined.iter_rows(named=True):
        predicted_list = row["predicted_items"]
        actual_set = set(row["actual_items"])
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(predicted_list[:k]):
            if item in actual_set:
                # rel = 1 for relevant items, 0 otherwise
                dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed
        
        # Calculate IDCG (ideal DCG)
        num_relevant = min(len(actual_set), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def mean_average_precision_at_k(
    predictions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    k: int,
    user_col: str = "customer_id",
    item_col: str = "item_id",
    score_col: str = "score",
) -> float:
    """Calculate Mean Average Precision@K metric.
    
    MAP@K is the mean of average precision scores for each user,
    where average precision considers both precision and ranking.
    
    Args:
        predictions: DataFrame with predictions (user, item, score).
        ground_truth: DataFrame with actual purchases (user, item).
        k: Number of top recommendations to consider.
        user_col: Name of the user ID column.
        item_col: Name of the item ID column.
        score_col: Name of the score/ranking column.
        
    Returns:
        Mean average precision@K across all users.
    """
    # Get top-K predictions per user
    top_k_preds = (
        predictions
        .sort([user_col, score_col], descending=[False, True])
        .group_by(user_col)
        .agg([
            pl.col(item_col).head(k).alias("predicted_items"),
        ])
    )
    
    # Get actual purchased items per user
    actual = (
        ground_truth
        .group_by(user_col)
        .agg([
            pl.col(item_col).alias("actual_items"),
        ])
    )
    
    # Join predictions with ground truth
    joined = top_k_preds.join(actual, on=user_col, how="inner")
    
    # Calculate AP for each user
    average_precisions = []
    for row in joined.iter_rows(named=True):
        predicted_list = row["predicted_items"]
        actual_set = set(row["actual_items"])
        
        if not actual_set:
            continue
        
        # Calculate average precision
        hits = 0
        precision_sum = 0.0
        
        for i, item in enumerate(predicted_list[:k]):
            if item in actual_set:
                hits += 1
                precision_at_i = hits / (i + 1)
                precision_sum += precision_at_i
        
        ap = precision_sum / min(len(actual_set), k) if actual_set else 0.0
        average_precisions.append(ap)
    
    return float(np.mean(average_precisions)) if average_precisions else 0.0


def evaluate_recommendations(
    predictions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    k_values: Sequence[int] = (5, 10, 20),
    user_col: str = "customer_id",
    item_col: str = "item_id",
    score_col: str = "score",
) -> Dict[str, Dict[int, float]]:
    """Evaluate recommendations using multiple metrics at different K values.
    
    Args:
        predictions: DataFrame with predictions (user, item, score).
        ground_truth: DataFrame with actual purchases (user, item).
        k_values: List of K values to evaluate.
        user_col: Name of the user ID column.
        item_col: Name of the item ID column.
        score_col: Name of the score/ranking column.
        
    Returns:
        Dictionary with metric names and their values at different K.
    """
    results = {
        "precision": {},
        "recall": {},
        "ndcg": {},
        "map": {},
    }
    
    for k in k_values:
        results["precision"][k] = precision_at_k(
            predictions, ground_truth, k, user_col, item_col, score_col
        )
        results["recall"][k] = recall_at_k(
            predictions, ground_truth, k, user_col, item_col, score_col
        )
        results["ndcg"][k] = ndcg_at_k(
            predictions, ground_truth, k, user_col, item_col, score_col
        )
        results["map"][k] = mean_average_precision_at_k(
            predictions, ground_truth, k, user_col, item_col, score_col
        )
    
    return results
