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


def _compute_recency_features(
    candidates: pl.LazyFrame,
    hist_txns: pl.LazyFrame,
    end_hist: datetime,
) -> pl.LazyFrame:
    """Compute recency features for each customer.
    
    Args:
        candidates: LazyFrame with (customer_id, item_id) pairs.
        hist_txns: Historical transactions.
        end_hist: End date of historical window.
        
    Returns:
        LazyFrame with (customer_id, days_since_last_purchase).
    """
    last_purchase = (
        hist_txns
        .group_by("customer_id")
        .agg(pl.col("created_date").max().alias("last_purchase_date"))
        .with_columns(
            ((pl.lit(end_hist).dt.date() - pl.col("last_purchase_date")).dt.total_days())
            .cast(pl.Int32)
            .alias("X4_days_since_last_purchase")
        )
        .select(["customer_id", "X4_days_since_last_purchase"])
    )
    return last_purchase


def _compute_frequency_features(
    candidates: pl.LazyFrame,
    hist_txns: pl.LazyFrame,
) -> pl.LazyFrame:
    """Compute frequency features for each customer.
    
    Args:
        candidates: LazyFrame with (customer_id, item_id) pairs.
        hist_txns: Historical transactions.
        
    Returns:
        LazyFrame with (customer_id, purchase_frequency_hist, is_power_user).
    """
    freq_features = (
        hist_txns
        .group_by("customer_id")
        .agg([
            pl.count().alias("num_purchases"),
            pl.col("created_date").n_unique().alias("days_active")
        ])
        .with_columns([
            (pl.col("num_purchases") / pl.col("days_active").clip(1))
            .alias("X5_purchase_frequency"),
            (pl.col("num_purchases") > 13).cast(pl.Int32).alias("X6_is_power_user")
        ])
        .select(["customer_id", "X5_purchase_frequency", "X6_is_power_user"])
    )
    return freq_features


def _compute_monetary_features(
    candidates: pl.LazyFrame,
    hist_txns: pl.LazyFrame,
) -> pl.LazyFrame:
    """Compute monetary/basket features for each customer.
    
    Args:
        candidates: LazyFrame with (customer_id, item_id) pairs.
        hist_txns: Historical transactions.
        
    Returns:
        LazyFrame with (customer_id, avg_items_per_purchase).
    """
    monetary_features = (
        hist_txns
        .group_by("customer_id")
        .agg([
            pl.col("item_id").n_unique().alias("total_unique_items"),
            pl.col("created_date").n_unique().alias("num_purchase_days")
        ])
        .with_columns(
            (pl.col("total_unique_items") / pl.col("num_purchase_days").clip(1))
            .alias("X7_avg_items_per_purchase")
        )
        .select(["customer_id", "X7_avg_items_per_purchase"])
    )
    return monetary_features


def _compute_brand_loyalty_features(
    candidates: pl.LazyFrame,
    hist_txns: pl.LazyFrame,
    items: pl.LazyFrame,
) -> pl.LazyFrame:
    """Compute brand loyalty features for each customer.
    
    Args:
        candidates: LazyFrame with (customer_id, item_id) pairs.
        hist_txns: Historical transactions.
        items: Item metadata.
        
    Returns:
        LazyFrame with (customer_id, top_brand_ratio, brand_diversity).
    """
    hist_with_brands = hist_txns.join(
        items.select(["item_id", "brand"]),
        on="item_id",
        how="left"
    )
    
    # Top brand ratio
    brand_counts = (
        hist_with_brands
        .group_by(["customer_id", "brand"])
        .agg(pl.count().alias("brand_count"))
    )
    
    total_purchases = (
        hist_with_brands
        .group_by("customer_id")
        .agg(pl.count().alias("total_purchases"))
    )
    
    top_brand = (
        brand_counts
        .sort(["customer_id", "brand_count"], descending=[False, True])
        .group_by("customer_id")
        .agg(pl.col("brand_count").first().alias("top_brand_count"))
    )
    
    brand_features = (
        top_brand
        .join(total_purchases, on="customer_id", how="left")
        .with_columns([
            (pl.col("top_brand_count") / pl.col("total_purchases"))
            .alias("X8_top_brand_ratio")
        ])
    )
    
    # Brand diversity
    brand_diversity = (
        hist_with_brands
        .group_by("customer_id")
        .agg(pl.col("brand").n_unique().alias("X9_brand_diversity"))
    )
    
    result = brand_features.join(
        brand_diversity,
        on="customer_id",
        how="left"
    ).select(["customer_id", "X8_top_brand_ratio", "X9_brand_diversity"])
    
    return result


def _compute_category_diversity_features(
    candidates: pl.LazyFrame,
    hist_txns: pl.LazyFrame,
    items: pl.LazyFrame,
) -> pl.LazyFrame:
    """Compute category diversity features for each customer.
    
    Args:
        candidates: LazyFrame with (customer_id, item_id) pairs.
        hist_txns: Historical transactions.
        items: Item metadata.
        
    Returns:
        LazyFrame with (customer_id, category_diversity_score).
    """
    hist_with_categories = hist_txns.join(
        items.select(["item_id", "category"]),
        on="item_id",
        how="left"
    )
    
    category_diversity = (
        hist_with_categories
        .group_by("customer_id")
        .agg([
            pl.col("category").n_unique().alias("unique_categories"),
            pl.count().alias("total_purchases")
        ])
        .with_columns(
            (pl.col("unique_categories") / pl.col("total_purchases"))
            .alias("X10_category_diversity_score")
        )
        .select(["customer_id", "X10_category_diversity_score"])
    )
    
    return category_diversity


def _compute_temporal_features(
    candidates: pl.LazyFrame,
    hist_txns: pl.LazyFrame,
) -> pl.LazyFrame:
    """Compute temporal pattern features for each customer.
    
    Args:
        candidates: LazyFrame with (customer_id, item_id) pairs.
        hist_txns: Historical transactions.
        
    Returns:
        LazyFrame with (customer_id, purchase_day_of_week_mode).
    """
    temporal_features = (
        hist_txns
        .with_columns(
            pl.col("created_date").dt.weekday().alias("day_of_week")
        )
        .group_by(["customer_id", "day_of_week"])
        .agg(pl.count().alias("day_count"))
        .sort(["customer_id", "day_count"], descending=[False, True])
        .group_by("customer_id")
        .agg(pl.col("day_of_week").first().alias("X11_purchase_day_mode"))
    )
    
    return temporal_features


def _compute_cold_start_features(
    candidates: pl.LazyFrame,
    hist_txns: pl.LazyFrame,
    items: pl.LazyFrame,
) -> pl.LazyFrame:
    """Compute cold start indicator features.
    
    Args:
        candidates: LazyFrame with (customer_id, item_id) pairs.
        hist_txns: Historical transactions.
        items: Item metadata.
        
    Returns:
        LazyFrame with (customer_id, is_new_customer, avg_item_popularity).
    """
    # Is new customer
    customer_purchase_count = (
        hist_txns
        .group_by("customer_id")
        .agg(pl.count().alias("num_purchases"))
        .with_columns(
            (pl.col("num_purchases") < 3).cast(pl.Int32).alias("X12_is_new_customer")
        )
        .select(["customer_id", "X12_is_new_customer"])
    )
    
    # Item popularity
    item_popularity = (
        hist_txns
        .group_by("item_id")
        .agg(pl.count().alias("item_popularity"))
    )
    
    # Average item popularity per customer
    hist_with_popularity = hist_txns.join(
        item_popularity,
        on="item_id",
        how="left"
    )
    
    avg_popularity = (
        hist_with_popularity
        .group_by("customer_id")
        .agg(pl.col("item_popularity").mean().alias("X13_avg_item_popularity"))
    )
    
    result = customer_purchase_count.join(
        avg_popularity,
        on="customer_id",
        how="left"
    ).with_columns(
        pl.col("X13_avg_item_popularity").fill_null(0)
    )
    
    return result


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
        Output columns: customer_id, item_id, X1-X13 (features), Y (label)
        Features: X1_brand_cnt_hist, X2_age_group_cnt_hist, X3_category_cnt_hist,
                 X4_days_since_last_purchase, X5_purchase_frequency, X6_is_power_user,
                 X7_avg_items_per_purchase, X8_top_brand_ratio, X9_brand_diversity,
                 X10_category_diversity_score, X11_purchase_day_mode, X12_is_new_customer,
                 X13_avg_item_popularity, Y
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
    
    # Build basic features for each candidate
    features = _build_candidate_features(candidates, hist_txns, items)
    
    # Build additional features (customer-level)
    recency_features = _compute_recency_features(candidates, hist_txns, end_hist)
    frequency_features = _compute_frequency_features(candidates, hist_txns)
    monetary_features = _compute_monetary_features(candidates, hist_txns)
    brand_loyalty_features = _compute_brand_loyalty_features(candidates, hist_txns, items)
    category_diversity_features = _compute_category_diversity_features(candidates, hist_txns, items)
    temporal_features = _compute_temporal_features(candidates, hist_txns)
    cold_start_features = _compute_cold_start_features(candidates, hist_txns, items)
    
    # Join all customer-level features using left joins to avoid suffix issues
    customer_features = (
        recency_features
        .join(frequency_features, on="customer_id", how="left")
        .join(monetary_features, on="customer_id", how="left")
        .join(brand_loyalty_features, on="customer_id", how="left")
        .join(category_diversity_features, on="customer_id", how="left")
        .join(temporal_features, on="customer_id", how="left")
        .join(cold_start_features, on="customer_id", how="left")
    )
    
    # Join with candidate features
    features_all = features.join(
        customer_features,
        on="customer_id",
        how="left"
    )
    
    # Fill nulls for all feature columns
    feature_cols = [
        "X4_days_since_last_purchase", "X5_purchase_frequency", "X6_is_power_user",
        "X7_avg_items_per_purchase", "X8_top_brand_ratio", "X9_brand_diversity",
        "X10_category_diversity_score", "X11_purchase_day_mode", "X12_is_new_customer",
        "X13_avg_item_popularity"
    ]
    
    features_all = features_all.with_columns([
        pl.col(col).fill_null(0) for col in feature_cols
    ])
    
    # Build labels
    labels = _build_labels_for_features(recent_txns)
    
    # Join features with labels
    result = features_all.join(
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

