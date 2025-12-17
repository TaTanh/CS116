"""Candidate generation strategies for recommender system."""

from datetime import datetime, timedelta
from typing import Literal, List

import polars as pl


def build_item_cooccurrence(
    transactions: pl.LazyFrame,
    key_cols: List[str],
) -> pl.LazyFrame:
    """Build item co-occurrence matrix from transactions.
    
    For each basket (identified by key_cols), finds all pairs of distinct items
    that appear together and counts their co-occurrences across all baskets.
    
    Args:
        transactions: LazyFrame with transaction data.
        key_cols: Column names that identify a basket (e.g., ["customer_id", "order_id"]).
        
    Returns:
        LazyFrame with columns: itemA, itemB, cooc_count
        where itemA < itemB (unordered pairs, no duplicates).
    """
    # Select basket keys and item_id
    baskets = transactions.select(key_cols + ["item_id"]).unique()
    
    # Self-join to get all pairs within each basket
    # Use aliases to distinguish the two sides
    pairs = baskets.join(
        baskets,
        on=key_cols,
        how="inner"
    )
    
    # Filter to get distinct pairs where itemA < itemB
    # This ensures each pair appears once and excludes self-pairs
    cooc = (
        pairs
        .filter(pl.col("item_id") < pl.col("item_id_right"))
        .group_by([
            pl.col("item_id").alias("itemA"),
            pl.col("item_id_right").alias("itemB")
        ])
        .agg(pl.count().alias("cooc_count"))
        .sort("cooc_count", descending=True)
    )
    
    return cooc


def generate_candidates_from_cooc(
    customer_hist_items: pl.LazyFrame,
    cooc: pl.LazyFrame,
    topn_per_customer: int = 200,
) -> pl.LazyFrame:
    """Generate candidate recommendations using item co-occurrence.
    
    For each item a customer has purchased, recommends items that frequently
    co-occur with it, ranked by co-occurrence count.
    
    Args:
        customer_hist_items: LazyFrame with columns (customer_id, item_id).
        cooc: LazyFrame with co-occurrence counts (itemA, itemB, cooc_count).
        topn_per_customer: Maximum number of candidates per customer.
        
    Returns:
        LazyFrame with columns (customer_id, item_id) for recommended items.
    """
    # Join customer items with co-occurrence in both directions
    # Direction 1: customer bought itemA, recommend itemB
    candidates_forward = (
        customer_hist_items
        .join(
            cooc.select([
                pl.col("itemA").alias("item_id"),
                pl.col("itemB").alias("candidate_item"),
                "cooc_count"
            ]),
            on="item_id",
            how="inner"
        )
        .select(["customer_id", "candidate_item", "cooc_count"])
    )
    
    # Direction 2: customer bought itemB, recommend itemA
    candidates_backward = (
        customer_hist_items
        .join(
            cooc.select([
                pl.col("itemB").alias("item_id"),
                pl.col("itemA").alias("candidate_item"),
                "cooc_count"
            ]),
            on="item_id",
            how="inner"
        )
        .select(["customer_id", "candidate_item", "cooc_count"])
    )
    
    # Combine both directions
    all_candidates = pl.concat([candidates_forward, candidates_backward])
    
    # Aggregate by (customer_id, candidate_item) - sum co-occurrence counts
    # if an item is recommended from multiple sources
    aggregated = (
        all_candidates
        .group_by(["customer_id", "candidate_item"])
        .agg(pl.col("cooc_count").sum().alias("total_cooc"))
    )
    
    # Rank by co-occurrence count and take top N per customer
    top_candidates = (
        aggregated
        .sort(["customer_id", "total_cooc"], descending=[False, True])
        .with_columns(
            pl.col("candidate_item").rank("dense").over("customer_id").alias("rank")
        )
        .filter(pl.col("rank") <= topn_per_customer)
        .select([
            "customer_id",
            pl.col("candidate_item").alias("item_id")
        ])
    )
    
    return top_candidates


def generate_candidates(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    observation_date: datetime,
    strategy: Literal["user_history", "popular_items", "category_based", "hybrid"] = "hybrid",
    lookback_days: int = 90,
    max_candidates_per_user: int = 100,
) -> pl.LazyFrame:
    """Generate candidate items for each user.
    
    Args:
        transactions: LazyFrame with transaction data.
        items: LazyFrame with item metadata.
        users: LazyFrame with user data.
        observation_date: Reference date for candidate generation.
        strategy: Candidate generation strategy.
        lookback_days: Number of days to look back for user history.
        max_candidates_per_user: Maximum number of candidates per user.
        
    Returns:
        LazyFrame with (customer_id, item_id) candidate pairs.
    """
    if strategy == "user_history":
        return _generate_user_history_candidates(
            transactions, observation_date, lookback_days, max_candidates_per_user
        )
    elif strategy == "popular_items":
        return _generate_popular_item_candidates(
            transactions, users, observation_date, lookback_days, max_candidates_per_user
        )
    elif strategy == "category_based":
        return _generate_category_based_candidates(
            transactions, items, observation_date, lookback_days, max_candidates_per_user
        )
    elif strategy == "hybrid":
        return _generate_hybrid_candidates(
            transactions, items, users, observation_date, lookback_days, max_candidates_per_user
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _generate_user_history_candidates(
    transactions: pl.LazyFrame,
    observation_date: datetime,
    lookback_days: int,
    max_candidates: int,
) -> pl.LazyFrame:
    """Generate candidates based on user's purchase history.
    
    Args:
        transactions: LazyFrame with transaction data.
        observation_date: Reference date.
        lookback_days: Number of days to look back.
        max_candidates: Maximum candidates per user.
        
    Returns:
        LazyFrame with candidate pairs.
    """
    start_date = observation_date - timedelta(days=lookback_days)
    
    # Get items purchased by each user in the lookback window
    candidates = (
        transactions
        .filter(
            (pl.col("created_at") >= start_date) &
            (pl.col("created_at") < observation_date)
        )
        .select(["customer_id", "item_id"])
        .unique()
    )
    
    # Limit candidates per user
    candidates = (
        candidates
        .with_columns(
            pl.col("item_id").rank("random").over("customer_id").alias("rank")
        )
        .filter(pl.col("rank") <= max_candidates)
        .select(["customer_id", "item_id"])
    )
    
    return candidates


def _generate_popular_item_candidates(
    transactions: pl.LazyFrame,
    users: pl.LazyFrame,
    observation_date: datetime,
    lookback_days: int,
    max_candidates: int,
) -> pl.LazyFrame:
    """Generate candidates from globally popular items.
    
    Args:
        transactions: LazyFrame with transaction data.
        users: LazyFrame with user data.
        observation_date: Reference date.
        lookback_days: Number of days to look back.
        max_candidates: Maximum candidates per user.
        
    Returns:
        LazyFrame with candidate pairs.
    """
    start_date = observation_date - timedelta(days=lookback_days)
    
    # Get top popular items
    popular_items = (
        transactions
        .filter(
            (pl.col("created_at") >= start_date) &
            (pl.col("created_at") < observation_date)
        )
        .group_by("item_id")
        .agg(pl.count().alias("popularity"))
        .sort("popularity", descending=True)
        .head(max_candidates)
        .select("item_id")
    )
    
    # Cross join with all users
    user_ids = users.select("customer_id")
    candidates = user_ids.join(popular_items, how="cross")
    
    return candidates


def _generate_category_based_candidates(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    observation_date: datetime,
    lookback_days: int,
    max_candidates: int,
) -> pl.LazyFrame:
    """Generate candidates based on user's preferred categories.
    
    Args:
        transactions: LazyFrame with transaction data.
        items: LazyFrame with item metadata.
        observation_date: Reference date.
        lookback_days: Number of days to look back.
        max_candidates: Maximum candidates per user.
        
    Returns:
        LazyFrame with candidate pairs.
    """
    start_date = observation_date - timedelta(days=lookback_days)
    
    # Join transactions with item categories
    txns_with_category = (
        transactions
        .filter(
            (pl.col("created_at") >= start_date) &
            (pl.col("created_at") < observation_date)
        )
        .join(items.select(["item_id", "category"]), on="item_id", how="left")
    )
    
    # Get user's top categories
    user_categories = (
        txns_with_category
        .group_by(["customer_id", "category"])
        .agg(pl.count().alias("category_count"))
        .sort("category_count", descending=True)
        .with_columns(
            pl.col("category").rank("dense").over("customer_id").alias("category_rank")
        )
        .filter(pl.col("category_rank") <= 3)  # Top 3 categories per user
        .select(["customer_id", "category"])
    )
    
    # Get popular items in those categories
    category_popular_items = (
        txns_with_category
        .join(items.select(["item_id", "category"]), on="item_id", how="left")
        .group_by(["category", "item_id"])
        .agg(pl.count().alias("item_popularity"))
        .sort("item_popularity", descending=True)
        .with_columns(
            pl.col("item_id").rank("dense").over("category").alias("item_rank")
        )
        .filter(pl.col("item_rank") <= (max_candidates // 3))
        .select(["category", "item_id"])
    )
    
    # Join user categories with category popular items
    candidates = user_categories.join(
        category_popular_items,
        on="category",
        how="left"
    ).select(["customer_id", "item_id"])
    
    # Limit candidates per user
    candidates = (
        candidates
        .with_columns(
            pl.col("item_id").rank("random").over("customer_id").alias("rank")
        )
        .filter(pl.col("rank") <= max_candidates)
        .select(["customer_id", "item_id"])
    )
    
    return candidates


def _generate_hybrid_candidates(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    observation_date: datetime,
    lookback_days: int,
    max_candidates: int,
) -> pl.LazyFrame:
    """Generate candidates using a hybrid approach.
    
    Combines user history, popular items, and category-based candidates.
    
    Args:
        transactions: LazyFrame with transaction data.
        items: LazyFrame with item metadata.
        users: LazyFrame with user data.
        observation_date: Reference date.
        lookback_days: Number of days to look back.
        max_candidates: Maximum candidates per user.
        
    Returns:
        LazyFrame with candidate pairs.
    """
    # Generate candidates from each strategy
    user_history = _generate_user_history_candidates(
        transactions, observation_date, lookback_days, max_candidates // 3
    )
    
    popular = _generate_popular_item_candidates(
        transactions, users, observation_date, lookback_days, max_candidates // 3
    )
    
    category_based = _generate_category_based_candidates(
        transactions, items, observation_date, lookback_days, max_candidates // 3
    )
    
    # Combine and deduplicate
    candidates = pl.concat([user_history, popular, category_based]).unique()
    
    # Limit candidates per user
    candidates = (
        candidates
        .with_columns(
            pl.col("item_id").rank("random").over("customer_id").alias("rank")
        )
        .filter(pl.col("rank") <= max_candidates)
        .select(["customer_id", "item_id"])
    )
    
    return candidates


def filter_already_purchased(
    candidates: pl.LazyFrame,
    transactions: pl.LazyFrame,
    observation_date: datetime,
    lookback_days: int = 30,
) -> pl.LazyFrame:
    """Filter out items already purchased recently.
    
    Args:
        candidates: LazyFrame with candidate pairs.
        transactions: LazyFrame with transaction data.
        observation_date: Reference date.
        lookback_days: Number of days to look back for filtering.
        
    Returns:
        Filtered LazyFrame with candidate pairs.
    """
    start_date = observation_date - timedelta(days=lookback_days)
    
    # Get recently purchased items
    recent_purchases = (
        transactions
        .filter(
            (pl.col("created_at") >= start_date) &
            (pl.col("created_at") < observation_date)
        )
        .select(["customer_id", "item_id"])
        .unique()
    )
    
    # Anti-join to remove already purchased items
    filtered_candidates = candidates.join(
        recent_purchases,
        on=["customer_id", "item_id"],
        how="anti"
    )
    
    return filtered_candidates
