"""Feature engineering for time-based purchase prediction."""

from datetime import datetime, timedelta
from typing import Optional, Union

import polars as pl


def add_baby_age_feature(
    users: pl.LazyFrame,
    ref_date: datetime,
) -> pl.LazyFrame:
    """Add baby age in months feature to users data.
    
    Computes age in months from date of birth to reference date using
    the formula: floor((ref_date - dob).days / 30.4375)
    
    Args:
        users: LazyFrame with columns (customer_id, date_of_birth).
        ref_date: Reference date for age calculation.
        
    Returns:
        LazyFrame with original columns plus age_in_month (int, null for null dob).
    """
    result = users.with_columns(
        (
            (pl.lit(ref_date).dt.date() - pl.col("date_of_birth")).dt.total_days() / 30.4375
        )
        .floor()
        .cast(pl.Int32)
        .alias("age_in_month")
    )
    
    return result


def build_item_segment_step1(items: pl.LazyFrame) -> pl.LazyFrame:
    """Build item segments based on category and age group.
    
    Segments items into "milk_step1" if category contains "sua" and 
    age_group is "Step 1", otherwise "other".
    
    Args:
        items: LazyFrame with columns (item_id, category, age_group).
        
    Returns:
        LazyFrame with columns (item_id, category, segment).
    """
    result = items.with_columns(
        pl.when(
            pl.col("category").str.contains("sua") & 
            (pl.col("age_group") == "Step 1")
        )
        .then(pl.lit("milk_step1"))
        .otherwise(pl.lit("other"))
        .alias("segment")
    ).select(["item_id", "category", "segment"])
    
    return result


def build_customer_segment_from_step1(
    transactions_hist: pl.LazyFrame,
    item_segment: pl.LazyFrame,
) -> pl.LazyFrame:
    """Assign customer segments based on most frequently purchased item segment.
    
    For each customer, determines their segment based on the most common
    segment they purchased in historical transactions. In case of ties,
    selects the segment from the most recent transaction.
    
    Args:
        transactions_hist: LazyFrame with columns (customer_id, item_id, created_date).
        item_segment: LazyFrame with columns (item_id, segment).
        
    Returns:
        LazyFrame with columns (customer_id, segment).
    """
    # Join transactions with item segments
    txns_with_segment = transactions_hist.join(
        item_segment.select(["item_id", "segment"]),
        on="item_id",
        how="left"
    )
    
    # Count segment frequency per customer and get latest date per segment
    segment_stats = (
        txns_with_segment
        .group_by(["customer_id", "segment"])
        .agg([
            pl.count().alias("segment_count"),
            pl.col("created_date").max().alias("latest_purchase")
        ])
    )
    
    # Rank segments by count (descending), then by latest_purchase (descending)
    # Pick rank 1 as the customer's segment
    customer_segments = (
        segment_stats
        .sort(["customer_id", "segment_count", "latest_purchase"], descending=[False, True, True])
        .with_columns(
            pl.col("segment").rank("first").over("customer_id").alias("rank")
        )
        .filter(pl.col("rank") == 1)
        .select(["customer_id", "segment"])
    )
    
    return customer_segments


def build_feature_label_table(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    begin_hist: datetime,
    end_hist: datetime,
    begin_recent: datetime,
    end_recent: datetime,
    candidates: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """Build feature and label table for training/prediction.
    
    Creates features from historical transactions in [begin_hist, end_hist)
    and labels from recent purchases in [begin_recent, end_recent).
    
    Args:
        transactions: LazyFrame with transaction data.
        items: LazyFrame with item metadata.
        users: LazyFrame with user data.
        begin_hist: Start of historical window (inclusive).
        end_hist: End of historical window (exclusive).
        begin_recent: Start of recent/label window (inclusive).
        end_recent: End of recent/label window (exclusive).
        candidates: Optional LazyFrame with (customer_id, item_id) pairs.
                   If None, candidates are generated automatically.
        
    Returns:
        LazyFrame with features and labels per (customer_id, item_id) pair.
        Output columns: customer_id, item_id, X1_brand_cnt_hist, 
                       X2_age_group_cnt_hist, X3_category_cnt_hist, Y
    """
    # Filter transactions for historical window
    hist_txns = transactions.filter(
        (pl.col("created_date") >= begin_hist) &
        (pl.col("created_date") < end_hist)
    )
    
    # Filter transactions for recent/label window
    recent_txns = transactions.filter(
        (pl.col("created_date") >= begin_recent) &
        (pl.col("created_date") < end_recent)
    )
    
    # Generate candidates if not provided
    if candidates is None:
        candidates = _generate_candidates_for_features(
            hist_txns, recent_txns, items
        )
    
    # Build features for each candidate
    features = _build_candidate_features(candidates, hist_txns, items)
    
    # Build labels
    labels = _build_labels_for_features(recent_txns)
    
    # Join features with labels
    result = features.join(
        labels,
        on=["customer_id", "item_id"],
        how="left"
    ).with_columns(
        pl.col("Y").fill_null(0)
    )
    
    return result


def _generate_candidates_for_features(
    hist_txns: pl.LazyFrame,
    recent_txns: pl.LazyFrame,
    items: pl.LazyFrame,
) -> pl.LazyFrame:
    """Generate candidate (customer_id, item_id) pairs.
    
    Combines three sources:
    1. All positives from recent transactions
    2. Top 50 globally popular items in hist for each customer
    3. Items sharing categories with customer's hist purchases (max 200 per customer)
    
    Args:
        hist_txns: Historical transactions.
        recent_txns: Recent transactions.
        items: Item metadata.
        
    Returns:
        LazyFrame with unique (customer_id, item_id) pairs.
    """
    # (a) All positives from recent
    positives = recent_txns.select(["customer_id", "item_id"]).unique()
    
    # (b) Top 50 popular items in hist for each customer
    # First get top 50 items globally
    top_items = (
        hist_txns
        .group_by("item_id")
        .agg(pl.count().alias("item_count"))
        .sort("item_count", descending=True)
        .head(50)
        .select("item_id")
    )
    
    # Get all customers from hist
    all_customers = hist_txns.select("customer_id").unique()
    
    # Cross join customers with top items
    popular_candidates = all_customers.join(top_items, how="cross")
    
    # (c) Category-based candidates
    # Join hist transactions with item categories
    hist_with_items = hist_txns.join(
        items.select(["item_id", "category"]),
        on="item_id",
        how="left"
    )
    
    # Get categories each customer bought
    customer_categories = (
        hist_with_items
        .select(["customer_id", "category"])
        .unique()
    )
    
    # Get all items by category
    items_by_category = items.select(["item_id", "category"])
    
    # Join to get candidate items
    category_candidates = (
        customer_categories
        .join(items_by_category, on="category", how="left")
        .select(["customer_id", "item_id"])
        .with_columns(
            pl.col("item_id").rank("random").over("customer_id").alias("rank")
        )
        .filter(pl.col("rank") <= 200)
        .select(["customer_id", "item_id"])
    )
    
    # Combine all candidates and deduplicate
    all_candidates = pl.concat([
        positives,
        popular_candidates,
        category_candidates
    ]).unique()
    
    return all_candidates


def _build_candidate_features(
    candidates: pl.LazyFrame,
    hist_txns: pl.LazyFrame,
    items: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build features for candidate pairs.
    
    For each (customer_id, item_id) pair, compute:
    - X1: count of items customer bought in hist with same brand
    - X2: count of items customer bought in hist with same age_group
    - X3: count of items customer bought in hist with same category
    
    Args:
        candidates: LazyFrame with (customer_id, item_id) pairs.
        hist_txns: Historical transactions.
        items: Item metadata.
        
    Returns:
        LazyFrame with features.
    """
    # Join candidates with item metadata to get brand, age_group, category
    candidates_with_attrs = candidates.join(
        items.select(["item_id", "brand", "age_group", "category"]),
        on="item_id",
        how="left"
    )
    
    # Join hist transactions with item metadata
    hist_with_items = hist_txns.join(
        items.select(["item_id", "brand", "age_group", "category"]),
        on="item_id",
        how="left"
    )
    
    # Count by brand for each customer
    customer_brand_counts = (
        hist_with_items
        .group_by(["customer_id", "brand"])
        .agg(pl.count().alias("brand_count"))
    )
    
    # Count by age_group for each customer
    customer_age_group_counts = (
        hist_with_items
        .group_by(["customer_id", "age_group"])
        .agg(pl.count().alias("age_group_count"))
    )
    
    # Count by category for each customer
    customer_category_counts = (
        hist_with_items
        .group_by(["customer_id", "category"])
        .agg(pl.count().alias("category_count"))
    )
    
    # Join counts back to candidates
    features = (
        candidates_with_attrs
        .join(
            customer_brand_counts,
            on=["customer_id", "brand"],
            how="left"
        )
        .join(
            customer_age_group_counts,
            on=["customer_id", "age_group"],
            how="left"
        )
        .join(
            customer_category_counts,
            on=["customer_id", "category"],
            how="left"
        )
        .select([
            "customer_id",
            "item_id",
            pl.col("brand_count").fill_null(0).alias("X1_brand_cnt_hist"),
            pl.col("age_group_count").fill_null(0).alias("X2_age_group_cnt_hist"),
            pl.col("category_count").fill_null(0).alias("X3_category_cnt_hist"),
        ])
    )
    
    return features


def _build_labels_for_features(recent_txns: pl.LazyFrame) -> pl.LazyFrame:
    """Build labels from recent transactions.
    
    Args:
        recent_txns: Recent transactions.
        
    Returns:
        LazyFrame with Y=1 for purchased items.
    """
    labels = (
        recent_txns
        .select(["customer_id", "item_id"])
        .unique()
        .with_columns(pl.lit(1).alias("Y"))
    )
    
    return labels

